"""Unit tests for the Pandera-based QA validation layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from qa_validator import (
    CAMERA_BATCH_SCHEMA,
    format_failure_alerts,
    is_valid_camera_id,
    save_quarantined_rows,
    validate_and_split_batch,
    validate_batch_dataframe,
)


def _make_row(
    *,
    camera_id: str = "camera_1",
    frame_id: str = "camera_1_b001_f00",
    brightness: float = 165.0,
    blur: float = 0.12,
    embedding_dim: int = config.EMBEDDING_DIM,
) -> dict:
    return {
        "timestamp": "2026-05-26T12:00:00+00:00",
        "camera_id": camera_id,
        "frame_id": frame_id,
        "brightness_avg": brightness,
        "blur_metric": blur,
        "embedding": np.zeros(embedding_dim, dtype=np.float64),
    }


class TestCameraBatchSchema:
    def test_valid_batch_passes_schema(self, sample_valid_batch: pd.DataFrame) -> None:
        validated = CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)
        assert len(validated) == len(sample_valid_batch)

    def test_brightness_below_minimum_fails(self, sample_valid_batch: pd.DataFrame) -> None:
        sample_valid_batch.loc[0, "brightness_avg"] = 5.0
        with pytest.raises(Exception):
            CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)

    def test_brightness_above_maximum_fails(self, sample_valid_batch: pd.DataFrame) -> None:
        sample_valid_batch.loc[0, "brightness_avg"] = 300.0
        with pytest.raises(Exception):
            CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)

    def test_negative_blur_fails(self, sample_valid_batch: pd.DataFrame) -> None:
        sample_valid_batch.loc[0, "blur_metric"] = -0.01
        with pytest.raises(Exception):
            CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)

    def test_invalid_camera_id_fails(self, sample_valid_batch: pd.DataFrame) -> None:
        sample_valid_batch.loc[0, "camera_id"] = "invalid_cam"
        with pytest.raises(Exception):
            CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)

    def test_wrong_embedding_dimension_fails(
        self,
        sample_valid_batch: pd.DataFrame,
    ) -> None:
        sample_valid_batch.at[0, "embedding"] = np.zeros(64, dtype=np.float64)
        with pytest.raises(Exception):
            CAMERA_BATCH_SCHEMA.validate(sample_valid_batch, lazy=True)


class TestValidateAndSplitBatch:
    def test_all_valid_rows_returned(self, sample_valid_batch: pd.DataFrame) -> None:
        valid_df, quarantined_df, alerts = validate_and_split_batch(sample_valid_batch)
        assert len(valid_df) == 2
        assert quarantined_df.empty
        assert alerts == []

    def test_invalid_brightness_row_is_quarantined(self) -> None:
        batch = pd.DataFrame(
            [
                _make_row(frame_id="camera_1_b001_f00", brightness=165.0),
                _make_row(frame_id="camera_1_b001_f01", brightness=3.0),
            ]
        )
        valid_df, quarantined_df, alerts = validate_and_split_batch(batch)

        assert len(valid_df) == 1
        assert len(quarantined_df) == 1
        assert quarantined_df.iloc[0]["frame_id"] == "camera_1_b001_f01"
        assert alerts
        assert any(
            rule.get("column") == "brightness_avg"
            for alert in alerts
            for rule in alert["failed_rules"]
        )

    def test_invalid_camera_id_is_quarantined(self) -> None:
        batch = pd.DataFrame([_make_row(camera_id="bad_camera")])
        _, quarantined_df, alerts = validate_and_split_batch(batch)

        assert len(quarantined_df) == 1
        assert any(
            rule.get("column") == "camera_id"
            for alert in alerts
            for rule in alert["failed_rules"]
        )

    def test_wrong_embedding_size_is_quarantined(self) -> None:
        batch = pd.DataFrame([_make_row(embedding_dim=32)])
        _, quarantined_df, alerts = validate_and_split_batch(batch)

        assert len(quarantined_df) == 1
        assert any(
            rule.get("column") == "embedding"
            for alert in alerts
            for rule in alert["failed_rules"]
        )

    def test_mixed_batch_splits_correctly(self) -> None:
        batch = pd.DataFrame(
            [
                _make_row(frame_id="camera_1_b001_f00", brightness=165.0),
                _make_row(frame_id="camera_1_b001_f01", brightness=2.0),
                _make_row(frame_id="camera_1_b001_f02", blur=-0.5),
            ]
        )
        valid_df, quarantined_df, alerts = validate_and_split_batch(batch)

        assert len(valid_df) == 1
        assert len(quarantined_df) == 2
        assert len(alerts) >= 2


class TestValidateBatchDataframe:
    def test_quarantine_files_are_written(
        self,
        quarantine_dir: Path,
    ) -> None:
        batch = pd.DataFrame([_make_row(brightness=1.0)])
        result = validate_batch_dataframe(
            batch,
            batch_round=6,
            quarantine_dir=quarantine_dir,
        )

        assert result.status == "failed"
        assert len(result.quarantined_df) == 1
        assert len(result.valid_df) == 0
        assert result.quarantine_paths
        assert any(path.endswith(".json") for path in result.quarantine_paths)
        assert any(path.endswith("_reason.json") for path in result.quarantine_paths)

    def test_valid_batch_does_not_quarantine(
        self,
        sample_valid_batch: pd.DataFrame,
        quarantine_dir: Path,
    ) -> None:
        result = validate_batch_dataframe(
            sample_valid_batch,
            batch_round=1,
            quarantine_dir=quarantine_dir,
        )

        assert result.status == "passed"
        assert len(result.quarantined_df) == 0
        assert len(result.valid_df) == 2
        assert result.quarantine_paths == []


class TestUtilityFunctions:
    def test_is_valid_camera_id(self) -> None:
        assert is_valid_camera_id("camera_1") is True
        assert is_valid_camera_id("camera_99") is True
        assert is_valid_camera_id("cam_1") is False
        assert is_valid_camera_id("camera_x") is False

    def test_format_failure_alerts_structure(self) -> None:
        failure_cases = pd.DataFrame(
            {
                "index": [0, 0],
                "column": ["brightness_avg", "blur_metric"],
                "check": ["in_range(10.0, 255.0)", "greater_than_or_equal_to(0.0)"],
                "failure_case": [2.0, -0.1],
            }
        )
        alerts = format_failure_alerts(failure_cases)

        assert len(alerts) == 1
        assert alerts[0]["row_index"] == 0
        assert len(alerts[0]["failed_rules"]) == 2

    def test_save_quarantined_rows_creates_reason_file(
        self,
        quarantine_dir: Path,
    ) -> None:
        batch = pd.DataFrame([_make_row(brightness=1.0)])
        alerts = [
            {
                "row_index": 0,
                "failed_rules": [
                    {
                        "column": "brightness_avg",
                        "check": "in_range(10.0, 255.0)",
                        "failure_case": 1.0,
                    }
                ],
            }
        ]
        paths = save_quarantined_rows(
            batch,
            alerts,
            batch_round=3,
            quarantine_dir=quarantine_dir,
        )

        reason_files = list(quarantine_dir.glob("*_reason.json"))
        assert len(reason_files) == 1
        assert len(paths) == 2
