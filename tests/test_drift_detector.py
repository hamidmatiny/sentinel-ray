"""Unit tests for the distributed drift detection engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import ray

import config
from drift_detector import (
    DriftAnalyzer,
    compute_drift_report,
    generate_baseline_data,
)


def _make_stream_batch(
    *,
    camera_id: str = "camera_1",
    num_rows: int = 4,
    brightness_mean: float = config.GOLDEN_BASELINE_BRIGHTNESS_MEAN,
    brightness_std: float = 2.0,
    blur_mean: float = config.GOLDEN_BASELINE_BLUR_MEAN,
    embedding_mean: float = config.EMBEDDING_MEAN,
    embedding_std: float = config.EMBEDDING_STD,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic QA-validated batch for drift testing."""
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(
        embedding_mean,
        embedding_std,
        (num_rows, config.EMBEDDING_DIM),
    )
    return pd.DataFrame(
        {
            "timestamp": ["2026-05-26T12:00:00+00:00"] * num_rows,
            "camera_id": [camera_id] * num_rows,
            "frame_id": [f"{camera_id}_test_f{index:02d}" for index in range(num_rows)],
            "brightness_avg": rng.normal(brightness_mean, brightness_std, num_rows),
            "blur_metric": rng.normal(blur_mean, 0.01, num_rows),
            "embedding": [row for row in embeddings],
        }
    )


class TestGenerateBaselineData:
    def test_returns_expected_number_of_rows(self) -> None:
        baseline = generate_baseline_data(num_samples=500)
        assert len(baseline) == 500

    def test_brightness_distribution_is_centered_on_golden_mean(self) -> None:
        baseline = generate_baseline_data(num_samples=5000)
        brightness_mean = baseline["brightness_avg"].mean()
        brightness_std = baseline["brightness_avg"].std()

        assert abs(brightness_mean - config.GOLDEN_BASELINE_BRIGHTNESS_MEAN) < 3.0
        assert abs(brightness_std - config.GOLDEN_BASELINE_BRIGHTNESS_STD) < 3.0

    def test_blur_distribution_is_near_expected_mean(self) -> None:
        baseline = generate_baseline_data(num_samples=5000)
        blur_mean = baseline["blur_metric"].mean()
        assert abs(blur_mean - config.GOLDEN_BASELINE_BLUR_MEAN) < 0.05

    def test_embeddings_are_128_dimensional(self) -> None:
        baseline = generate_baseline_data(num_samples=10)
        for embedding in baseline["embedding"]:
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (config.EMBEDDING_DIM,)


class TestComputeDriftReport:
    def test_baseline_like_window_does_not_trigger_drift(self) -> None:
        baseline = generate_baseline_data(num_samples=1000)
        window = baseline.sample(n=config.DRIFT_WINDOW_SIZE, random_state=1).copy()

        report = compute_drift_report(window, baseline, camera_id="camera_1")

        assert report["status"] == "analyzed"
        assert report["drift_detected"] is False
        assert report["features"]["brightness_avg"]["p_value"] >= config.DRIFT_ALPHA
        assert report["features"]["blur_metric"]["p_value"] >= config.DRIFT_ALPHA

    def test_shifted_brightness_window_triggers_drift(self) -> None:
        baseline = generate_baseline_data(num_samples=1000)
        shifted_window = _make_stream_batch(
            num_rows=config.DRIFT_WINDOW_SIZE,
            brightness_mean=80.0,
            brightness_std=1.0,
            seed=2,
        )

        report = compute_drift_report(shifted_window, baseline, camera_id="camera_1")

        assert report["drift_detected"] is True
        assert bool(report["features"]["brightness_avg"]["drift_detected"])
        assert report["features"]["brightness_avg"]["p_value"] < config.DRIFT_ALPHA

    def test_shifted_embedding_window_triggers_drift(self) -> None:
        baseline = generate_baseline_data(num_samples=1000)
        shifted_window = _make_stream_batch(
            num_rows=config.DRIFT_WINDOW_SIZE,
            embedding_mean=4.0,
            embedding_std=1.5,
            seed=3,
        )

        report = compute_drift_report(shifted_window, baseline, camera_id="camera_1")

        assert report["drift_detected"] is True
        assert bool(report["features"]["embedding"]["drift_detected"])
        assert report["features"]["embedding"]["p_value"] < config.DRIFT_ALPHA

    def test_empty_window_is_skipped_gracefully(self) -> None:
        baseline = generate_baseline_data(num_samples=100)
        report = compute_drift_report(
            pd.DataFrame(),
            baseline,
            camera_id="camera_1",
        )

        assert report["status"] == "skipped"
        assert report["drift_detected"] is False


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialize a local Ray cluster for actor integration tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestDriftAnalyzerActor:
    def test_empty_input_is_skipped(self, ray_cluster) -> None:
        analyzer = DriftAnalyzer.remote()
        report = ray.get(analyzer.accumulate_and_analyze.remote("camera_1", pd.DataFrame()))

        assert report["status"] == "skipped"
        assert report["reason"] == "empty_input"
        assert report["drift_detected"] is False

    def test_buffering_status_before_window_is_full(self, ray_cluster) -> None:
        analyzer = DriftAnalyzer.remote()
        batch = _make_stream_batch(num_rows=4, seed=4)
        report = ray.get(analyzer.accumulate_and_analyze.remote("camera_1", batch))

        assert report["status"] == "buffering"
        assert report["buffer_size"] == 4
        assert report["drift_detected"] is False

    def test_intentional_shift_is_flagged_after_window_fills(self, ray_cluster) -> None:
        analyzer = DriftAnalyzer.remote()

        healthy_batch = _make_stream_batch(
            num_rows=10,
            brightness_mean=config.GOLDEN_BASELINE_BRIGHTNESS_MEAN,
            seed=5,
        )
        shifted_batch = _make_stream_batch(
            num_rows=10,
            brightness_mean=70.0,
            brightness_std=1.0,
            seed=6,
        )

        ray.get(analyzer.accumulate_and_analyze.remote("camera_1", healthy_batch))
        report = ray.get(analyzer.accumulate_and_analyze.remote("camera_1", shifted_batch))

        assert report["status"] == "analyzed"
        assert report["window_size"] == config.DRIFT_WINDOW_SIZE
        assert report["drift_detected"] is True
        assert report["features"]["brightness_avg"]["p_value"] < config.DRIFT_ALPHA
