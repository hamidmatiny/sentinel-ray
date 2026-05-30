"""Shared pytest fixtures for QA validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config


@pytest.fixture
def sample_valid_batch() -> pd.DataFrame:
    """Return a fully valid camera batch."""
    embeddings = [
        np.random.default_rng(0).normal(0.0, 1.0, config.EMBEDDING_DIM)
        for _ in range(2)
    ]
    return pd.DataFrame(
        {
            "timestamp": ["2026-05-26T12:00:00+00:00", "2026-05-26T12:00:01+00:00"],
            "camera_id": ["camera_1", "camera_1"],
            "frame_id": ["camera_1_b001_f00", "camera_1_b001_f01"],
            "brightness_avg": [165.0, 170.5],
            "blur_metric": [0.12, 0.15],
            "embedding": embeddings,
        }
    )


@pytest.fixture
def quarantine_dir(tmp_path: Path) -> Path:
    """Isolated quarantine directory for filesystem tests."""
    target = tmp_path / "quarantine"
    target.mkdir()
    return target
