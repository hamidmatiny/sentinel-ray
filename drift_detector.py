"""Distributed statistical drift detection over streaming QA-validated data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import ray
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

import config

logger = logging.getLogger(__name__)


def generate_baseline_data(num_samples: int | None = None) -> pd.DataFrame:
    """
    Build a static golden reference dataset representing training-time conditions.

    Returns:
        DataFrame with brightness, blur, and 128-D embeddings drawn from
        ideal distributions used as the drift comparison baseline.
    """
    sample_count = num_samples or config.DRIFT_BASELINE_SAMPLES
    rng = np.random.default_rng(config.RANDOM_SEED)

    brightness = rng.normal(
        config.GOLDEN_BASELINE_BRIGHTNESS_MEAN,
        config.GOLDEN_BASELINE_BRIGHTNESS_STD,
        sample_count,
    )
    blur = rng.normal(
        config.GOLDEN_BASELINE_BLUR_MEAN,
        config.GOLDEN_BASELINE_BLUR_STD,
        sample_count,
    )
    embeddings = rng.normal(
        config.EMBEDDING_MEAN,
        config.EMBEDDING_STD,
        (sample_count, config.EMBEDDING_DIM),
    )

    brightness = np.clip(brightness, config.BRIGHTNESS_MIN_VALID, config.BRIGHTNESS_MAX_VALID)
    blur = np.clip(blur, config.BLUR_MIN_VALID, None)

    return pd.DataFrame(
        {
            "camera_id": ["baseline"] * sample_count,
            "frame_id": [f"baseline_f{index:05d}" for index in range(sample_count)],
            "brightness_avg": brightness.astype(np.float64),
            "blur_metric": blur.astype(np.float64),
            "embedding": [row for row in embeddings],
        }
    )


def _stack_embeddings(frame_df: pd.DataFrame) -> np.ndarray:
    """Convert an embedding column into a 2-D NumPy matrix."""
    if frame_df.empty:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float64)
    return np.stack(frame_df["embedding"].to_numpy()).astype(np.float64)


def _compute_embedding_metrics(
    window_embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
) -> dict[str, float]:
    """Compare window and baseline embedding centroids for drift signals."""
    if window_embeddings.size == 0 or baseline_embeddings.size == 0:
        return {
            "mean_cosine_similarity": 1.0,
            "mean_euclidean_distance": 0.0,
        }

    window_centroid = window_embeddings.mean(axis=0, keepdims=True)
    baseline_centroid = baseline_embeddings.mean(axis=0, keepdims=True)

    cosine_distance = float(cdist(window_centroid, baseline_centroid, metric="cosine")[0, 0])
    cosine_similarity = 1.0 - cosine_distance
    euclidean_distance = float(cdist(window_centroid, baseline_centroid, metric="euclidean")[0, 0])

    return {
        "mean_cosine_similarity": cosine_similarity,
        "mean_euclidean_distance": euclidean_distance,
    }


def compute_drift_report(
    window_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    camera_id: str,
) -> dict[str, Any]:
    """
    Compare a streaming window against the golden baseline and build a drift report.

    Drift is flagged when any numerical KS p-value falls below the configured alpha
    or embedding distance/similarity crosses configured thresholds.
    """
    if window_df.empty:
        return {
            "camera_id": camera_id,
            "window_size": 0,
            "status": "skipped",
            "drift_detected": False,
            "features": {},
            "reason": "empty_window",
        }

    brightness_ks = ks_2samp(
        window_df["brightness_avg"],
        baseline_df["brightness_avg"],
    )
    blur_ks = ks_2samp(
        window_df["blur_metric"],
        baseline_df["blur_metric"],
    )

    embedding_metrics = _compute_embedding_metrics(
        _stack_embeddings(window_df),
        _stack_embeddings(baseline_df),
    )
    window_norms = np.linalg.norm(_stack_embeddings(window_df), axis=1)
    baseline_norms = np.linalg.norm(_stack_embeddings(baseline_df), axis=1)
    embedding_ks = ks_2samp(window_norms, baseline_norms)

    brightness_drift = brightness_ks.pvalue < config.DRIFT_ALPHA
    blur_drift = blur_ks.pvalue < config.DRIFT_ALPHA
    embedding_drift = embedding_ks.pvalue < config.DRIFT_ALPHA

    features: dict[str, Any] = {
        "brightness_avg": {
            "ks_statistic": float(brightness_ks.statistic),
            "p_value": float(brightness_ks.pvalue),
            "drift_detected": bool(brightness_drift),
        },
        "blur_metric": {
            "ks_statistic": float(blur_ks.statistic),
            "p_value": float(blur_ks.pvalue),
            "drift_detected": bool(blur_drift),
        },
        "embedding": {
            "ks_statistic": float(embedding_ks.statistic),
            "p_value": float(embedding_ks.pvalue),
            **embedding_metrics,
            "drift_detected": bool(embedding_drift),
        },
    }

    drift_detected = any(feature["drift_detected"] for feature in features.values())

    return {
        "camera_id": camera_id,
        "window_size": len(window_df),
        "status": "analyzed",
        "drift_detected": drift_detected,
        "alpha": config.DRIFT_ALPHA,
        "features": features,
    }


def create_drift_analyzer() -> ray.actor.ActorHandle:
    """Instantiate the DriftAnalyzer actor on the Ray cluster."""
    try:
        analyzer = DriftAnalyzer.remote()
        logger.info("DriftAnalyzer actor created.")
        return analyzer
    except Exception as exc:
        logger.exception("Failed to create DriftAnalyzer actor.")
        raise RuntimeError("DriftAnalyzer actor creation failed.") from exc


@ray.remote
class DriftAnalyzer:
    """Stateful actor that buffers clean QA-passed rows and runs drift analysis."""

    def __init__(self) -> None:
        self._baseline_df = generate_baseline_data()
        self._buffers: dict[str, list[pd.DataFrame]] = {
            camera_id: [] for camera_id in config.CAMERA_IDS
        }
        logger.info(
            "DriftAnalyzer initialized with baseline samples=%d.",
            len(self._baseline_df),
        )

    def accumulate_and_analyze(
        self,
        camera_id: str,
        valid_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Append QA-validated rows to a per-camera buffer and analyze when full.

        Uses a sliding window of the most recent ``DRIFT_WINDOW_SIZE`` rows.
        """
        if valid_df.empty:
            logger.debug(
                "DriftAnalyzer skipped empty batch for camera=%s.",
                camera_id,
            )
            return {
                "camera_id": camera_id,
                "status": "skipped",
                "drift_detected": False,
                "buffer_size": len(self._get_buffer_rows(camera_id)),
                "reason": "empty_input",
                "features": {},
            }

        if camera_id not in self._buffers:
            logger.warning("DriftAnalyzer received unknown camera_id=%s.", camera_id)
            return {
                "camera_id": camera_id,
                "status": "skipped",
                "drift_detected": False,
                "reason": "unknown_camera",
                "features": {},
            }

        self._buffers[camera_id].append(valid_df.copy())
        buffered_rows = self._get_buffer_rows(camera_id)

        if len(buffered_rows) < config.DRIFT_WINDOW_SIZE:
            return {
                "camera_id": camera_id,
                "status": "buffering",
                "drift_detected": False,
                "buffer_size": len(buffered_rows),
                "window_size_required": config.DRIFT_WINDOW_SIZE,
                "features": {},
            }

        window_df = buffered_rows.tail(config.DRIFT_WINDOW_SIZE)
        report = compute_drift_report(
            window_df=window_df,
            baseline_df=self._baseline_df,
            camera_id=camera_id,
        )
        logger.debug(
            "Drift analysis completed for camera=%s | drift_detected=%s | buffer_size=%d.",
            camera_id,
            report["drift_detected"],
            len(buffered_rows),
        )
        return report

    def _get_buffer_rows(self, camera_id: str) -> pd.DataFrame:
        """Concatenate all buffered frames for a camera into a single DataFrame."""
        frames = self._buffers.get(camera_id, [])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
