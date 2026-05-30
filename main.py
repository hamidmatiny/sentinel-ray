"""Entry point: ingestion stream routed through distributed QA validation."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd

import config
from ingestion_engine import (
    create_data_streamer,
    fetch_batches_concurrently,
    initialize_ray_cluster,
)
from qa_validator import create_qa_worker_pool, validate_batches_concurrently


def configure_logging() -> None:
    """Configure structured console logging for the pipeline."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def summarize_batch(batch_df: pd.DataFrame) -> dict[str, Any]:
    """Compute compact telemetry summary for console output."""
    if batch_df.empty:
        return {
            "camera_id": "unknown",
            "frame_count": 0,
            "frame_ids": [],
            "brightness_avg_mean": 0.0,
            "brightness_avg_min": 0.0,
            "blur_metric_mean": 0.0,
            "embedding_mean": 0.0,
            "embedding_std": 0.0,
        }

    embeddings = np.stack(batch_df["embedding"].to_numpy())
    return {
        "camera_id": batch_df["camera_id"].iloc[0],
        "frame_count": len(batch_df),
        "frame_ids": batch_df["frame_id"].tolist(),
        "brightness_avg_mean": float(batch_df["brightness_avg"].mean()),
        "brightness_avg_min": float(batch_df["brightness_avg"].min()),
        "blur_metric_mean": float(batch_df["blur_metric"].mean()),
        "embedding_mean": float(embeddings.mean()),
        "embedding_std": float(embeddings.std()),
    }


def log_qa_results(batch_index: int, qa_results: list[dict[str, Any]]) -> None:
    """Log QA validation outcomes for a batch round."""
    logger = logging.getLogger(__name__)
    for result in qa_results:
        logger.info(
            (
                "QA RESULT | batch_round=%d | camera=%s | status=%s | "
                "valid=%d | quarantined=%d"
            ),
            batch_index,
            result["camera_id"],
            result["status"],
            result["valid_count"],
            result["quarantined_count"],
        )
        if result.get("error_message"):
            logger.error(
                "QA ERROR | camera=%s | batch_round=%d | error=%s",
                result["camera_id"],
                batch_index,
                result["error_message"],
            )


def log_batch_telemetry(
    batch_index: int,
    summaries: list[dict[str, Any]],
    qa_results: list[dict[str, Any]],
) -> None:
    """Emit structured stream telemetry for validated batches only."""
    logger = logging.getLogger(__name__)
    logger.info("=== Batch round %d / %d ===", batch_index, config.TOTAL_BATCHES)

    log_qa_results(batch_index, qa_results)

    for summary in summaries:
        if summary["frame_count"] == 0:
            logger.info(
                "camera=%s | no valid frames passed QA (all quarantined).",
                summary["camera_id"],
            )
            continue

        logger.info(
            (
                "camera=%s | frames=%d | brightness_mean=%.2f | "
                "brightness_min=%.2f | blur_mean=%.4f | "
                "embedding_mean=%.4f | embedding_std=%.4f"
            ),
            summary["camera_id"],
            summary["frame_count"],
            summary["brightness_avg_mean"],
            summary["brightness_avg_min"],
            summary["blur_metric_mean"],
            summary["embedding_mean"],
            summary["embedding_std"],
        )
        logger.debug("frame_ids=%s", summary["frame_ids"])


def run_ingestion_pipeline() -> None:
    """Initialize Ray, validate batches via QA workers, then log telemetry."""
    logger = logging.getLogger(__name__)
    streamer = None

    try:
        initialize_ray_cluster()
        streamer = create_data_streamer()
        qa_workers = create_qa_worker_pool()

        logger.info(
            "Starting ingestion with QA gate: %d batch rounds x %d cameras.",
            config.TOTAL_BATCHES,
            config.NUM_CAMERAS,
        )

        total_quarantined = 0

        for batch_index in range(1, config.TOTAL_BATCHES + 1):
            raw_batches = fetch_batches_concurrently(streamer)
            qa_results = validate_batches_concurrently(
                qa_workers,
                raw_batches,
                batch_round=batch_index,
            )

            valid_batches = [result["valid_df"] for result in qa_results]
            summaries = []
            for result, valid_df in zip(qa_results, valid_batches):
                summary = summarize_batch(valid_df)
                if summary["frame_count"] == 0:
                    summary["camera_id"] = result["camera_id"]
                summaries.append(summary)
            log_batch_telemetry(batch_index, summaries, qa_results)

            total_quarantined += sum(
                result["quarantined_count"] for result in qa_results
            )

        logger.info(
            "Ingestion pipeline completed. total_quarantined_rows=%d",
            total_quarantined,
        )

    except Exception:
        logger.exception("Ingestion pipeline failed.")
        raise
    finally:
        if streamer is not None:
            logger.debug("DataStreamer actor session ended.")


def main() -> None:
    """Application entry point."""
    configure_logging()
    run_ingestion_pipeline()


if __name__ == "__main__":
    main()
