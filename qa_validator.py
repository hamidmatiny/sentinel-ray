"""Distributed QA validation layer using Pandera schema checks."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from pandera import Check, Column, DataFrameSchema
from pandera.errors import SchemaErrors

import config

logger = logging.getLogger(__name__)


def _brightness_valid(value: object) -> bool:
    """Return True when brightness is within the configured valid range."""
    return (
        isinstance(value, (int, float, np.floating, np.integer))
        and config.BRIGHTNESS_MIN_VALID <= float(value) <= config.BRIGHTNESS_MAX_VALID
    )


def _blur_valid(value: object) -> bool:
    """Return True when blur is non-negative."""
    return (
        isinstance(value, (int, float, np.floating, np.integer))
        and float(value) >= config.BLUR_MIN_VALID
    )


def _camera_id_valid(value: object) -> bool:
    """Return True when camera_id matches the configured pattern."""
    return isinstance(value, str) and bool(re.fullmatch(config.CAMERA_ID_PATTERN, value))


def _embedding_has_expected_dim(embedding: object) -> bool:
    """Return True when the embedding is a 128-dimensional vector."""
    return (
        isinstance(embedding, np.ndarray)
        and embedding.ndim == 1
        and embedding.size == config.EMBEDDING_DIM
    )


CAMERA_BATCH_SCHEMA = DataFrameSchema(
    {
        "timestamp": Column(str, nullable=False),
        "camera_id": Column(
            str,
            checks=Check(_camera_id_valid, element_wise=True),
            nullable=False,
        ),
        "frame_id": Column(str, nullable=False),
        "brightness_avg": Column(
            float,
            checks=Check(_brightness_valid, element_wise=True),
            nullable=False,
        ),
        "blur_metric": Column(
            float,
            checks=Check(_blur_valid, element_wise=True),
            nullable=False,
        ),
        "embedding": Column(
            object,
            checks=Check(_embedding_has_expected_dim, element_wise=True),
            nullable=False,
        ),
    },
    strict=False,
    coerce=True,
)


@dataclass
class ValidationResult:
    """Outcome of validating a single camera batch."""

    camera_id: str
    batch_round: int
    status: str
    valid_df: pd.DataFrame
    quarantined_df: pd.DataFrame
    alerts: list[dict[str, Any]] = field(default_factory=list)
    quarantine_paths: list[str] = field(default_factory=list)
    error_message: str | None = None


def format_failure_alerts(failure_cases: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert Pandera failure cases into structured alert payloads."""
    alerts: list[dict[str, Any]] = []
    if failure_cases.empty:
        return alerts

    grouped = failure_cases.groupby("index", dropna=False)
    for row_index, group in grouped:
        rules: list[dict[str, Any]] = []
        for _, failure in group.iterrows():
            rules.append(
                {
                    "column": failure.get("column"),
                    "check": failure.get("check"),
                    "failure_case": failure.get("failure_case"),
                }
            )
        alerts.append(
            {
                "row_index": None if pd.isna(row_index) else int(row_index),
                "failed_rules": rules,
            }
        )
    return alerts


def validate_and_split_batch(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """
    Validate a batch against the QA schema and split valid vs quarantined rows.

    Returns:
        Tuple of (valid_df, quarantined_df, alerts).
    """
    if batch_df.empty:
        return batch_df.copy(), batch_df.iloc[0:0].copy(), []

    try:
        valid_df = CAMERA_BATCH_SCHEMA.validate(batch_df, lazy=True)
        return valid_df, batch_df.iloc[0:0].copy(), []
    except SchemaErrors as exc:
        failure_cases = exc.failure_cases
        alerts = format_failure_alerts(failure_cases)

        failed_indices: set[int] = set()
        if "index" in failure_cases.columns:
            for raw_index in failure_cases["index"].dropna().unique():
                failed_indices.add(int(raw_index))

        quarantined_df = batch_df.loc[sorted(failed_indices)].copy()
        valid_df = batch_df.drop(index=sorted(failed_indices), errors="ignore").copy()
        return valid_df, quarantined_df, alerts


def save_quarantined_rows(
    quarantined_df: pd.DataFrame,
    alerts: list[dict[str, Any]],
    batch_round: int,
    quarantine_dir: Path | None = None,
) -> list[str]:
    """Persist failed rows and validation metadata to the quarantine directory."""
    target_dir = quarantine_dir or config.QUARANTINE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    alerts_by_index: dict[int | None, list[dict[str, Any]]] = {}
    for alert in alerts:
        alerts_by_index.setdefault(alert.get("row_index"), []).extend(
            alert.get("failed_rules", [])
        )

    saved_paths: list[str] = []
    for row_index, row in quarantined_df.iterrows():
        frame_id = str(row["frame_id"])
        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base_name = f"{row['camera_id']}_round{batch_round:03d}_{frame_id}_{timestamp_slug}"

        row_path = target_dir / f"{base_name}.json"
        reason_path = target_dir / f"{base_name}_reason.json"

        embedding = row["embedding"]
        row_payload = {
            "timestamp": row["timestamp"],
            "camera_id": row["camera_id"],
            "frame_id": frame_id,
            "brightness_avg": float(row["brightness_avg"]),
            "blur_metric": float(row["blur_metric"]),
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        }
        row_path.write_text(json.dumps(row_payload, indent=2), encoding="utf-8")

        reason_payload = {
            "batch_round": batch_round,
            "camera_id": row["camera_id"],
            "frame_id": frame_id,
            "row_index": int(row_index),
            "failed_rules": alerts_by_index.get(int(row_index), alerts_by_index.get(None, [])),
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        reason_path.write_text(json.dumps(reason_payload, indent=2), encoding="utf-8")

        saved_paths.extend([str(row_path), str(reason_path)])

    return saved_paths


def log_validation_alerts(
    camera_id: str,
    batch_round: int,
    alerts: list[dict[str, Any]],
    quarantined_count: int,
) -> None:
    """Emit a detailed alert when validation rules fail."""
    if quarantined_count == 0:
        return

    logger.warning(
        "QA ALERT | camera=%s | batch_round=%d | quarantined_rows=%d",
        camera_id,
        batch_round,
        quarantined_count,
    )
    for alert in alerts:
        for rule in alert.get("failed_rules", []):
            logger.warning(
                (
                    "QA RULE FAILURE | camera=%s | batch_round=%d | "
                    "row_index=%s | column=%s | check=%s | failure_case=%s"
                ),
                camera_id,
                batch_round,
                alert.get("row_index"),
                rule.get("column"),
                rule.get("check"),
                rule.get("failure_case"),
            )


def validate_batch_dataframe(
    batch_df: pd.DataFrame,
    batch_round: int,
    quarantine_dir: Path | None = None,
) -> ValidationResult:
    """
    Run schema validation, quarantine failures, and return a structured result.

    Validation failures are handled gracefully; unexpected errors are captured
    without propagating to the Ray runtime.
    """
    camera_id = "unknown"
    if not batch_df.empty and "camera_id" in batch_df.columns:
        camera_id = str(batch_df["camera_id"].iloc[0])

    try:
        valid_df, quarantined_df, alerts = validate_and_split_batch(batch_df)
        quarantine_paths: list[str] = []

        if not quarantined_df.empty:
            log_validation_alerts(camera_id, batch_round, alerts, len(quarantined_df))
            quarantine_paths = save_quarantined_rows(
                quarantined_df,
                alerts,
                batch_round,
                quarantine_dir=quarantine_dir,
            )

        status = "passed" if quarantined_df.empty else "partial_failure"
        if valid_df.empty and not quarantined_df.empty:
            status = "failed"

        return ValidationResult(
            camera_id=camera_id,
            batch_round=batch_round,
            status=status,
            valid_df=valid_df,
            quarantined_df=quarantined_df,
            alerts=alerts,
            quarantine_paths=quarantine_paths,
        )
    except Exception as exc:
        logger.exception(
            "Unexpected QA validation error for camera=%s batch_round=%d.",
            camera_id,
            batch_round,
        )
        return ValidationResult(
            camera_id=camera_id,
            batch_round=batch_round,
            status="error",
            valid_df=batch_df.iloc[0:0].copy(),
            quarantined_df=batch_df.copy(),
            alerts=[],
            error_message=str(exc),
        )


@ray.remote
class QAValidationWorker:
    """Ray actor that validates incoming stream batches and quarantines failures."""

    def __init__(self, quarantine_dir: str | None = None) -> None:
        self._quarantine_dir = Path(quarantine_dir or config.QUARANTINE_DIR)

    def validate_batch(
        self,
        batch_df: pd.DataFrame,
        batch_round: int,
    ) -> dict[str, Any]:
        """Validate a batch and return a Ray-serializable result dictionary."""
        result = validate_batch_dataframe(
            batch_df,
            batch_round=batch_round,
            quarantine_dir=self._quarantine_dir,
        )
        return {
            "camera_id": result.camera_id,
            "batch_round": result.batch_round,
            "status": result.status,
            "valid_df": result.valid_df,
            "quarantined_df": result.quarantined_df,
            "alerts": result.alerts,
            "quarantine_paths": result.quarantine_paths,
            "valid_count": len(result.valid_df),
            "quarantined_count": len(result.quarantined_df),
            "error_message": result.error_message,
        }


def create_qa_worker_pool(
    pool_size: int | None = None,
    quarantine_dir: Path | None = None,
) -> list[ray.actor.ActorHandle]:
    """Create a pool of QAValidationWorker actors for parallel validation."""
    size = pool_size or config.QA_WORKER_POOL_SIZE
    quarantine_path = str(quarantine_dir or config.QUARANTINE_DIR)
    try:
        workers = [
            QAValidationWorker.remote(quarantine_dir=quarantine_path)
            for _ in range(size)
        ]
        logger.info("QAValidationWorker pool created (size=%d).", size)
        return workers
    except Exception as exc:
        logger.exception("Failed to create QAValidationWorker pool.")
        raise RuntimeError("QAValidationWorker pool creation failed.") from exc


def validate_batches_concurrently(
    workers: list[ray.actor.ActorHandle],
    batches: list[pd.DataFrame],
    batch_round: int,
) -> list[dict[str, Any]]:
    """
    Route camera batches through the QA worker pool before downstream processing.

    Each batch is assigned to a worker in round-robin order and resolved via ray.get().
    """
    if not batches:
        return []

    if not workers:
        raise ValueError("At least one QAValidationWorker is required.")

    try:
        futures = [
            workers[index % len(workers)].validate_batch.remote(batch_df, batch_round)
            for index, batch_df in enumerate(batches)
        ]
        return ray.get(futures)
    except Exception as exc:
        logger.exception("Concurrent QA validation failed for batch_round=%d.", batch_round)
        raise RuntimeError("QA validation cluster failed.") from exc


def is_valid_camera_id(camera_id: str) -> bool:
    """Return True when camera_id matches the configured pattern."""
    return bool(re.fullmatch(config.CAMERA_ID_PATTERN, camera_id))
