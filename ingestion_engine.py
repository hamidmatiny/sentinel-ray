"""Ray Core ingestion engine with a stateful multi-camera data streamer."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import ray

import config

logger = logging.getLogger(__name__)


def initialize_ray_cluster() -> None:
    """Initialize a local Ray cluster if one is not already running."""
    try:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=config.RAY_NUM_CPUS,
            logging_level=logging.WARNING,
        )
        logger.info(
            "Ray cluster initialized (CPUs=%d, nodes=%d).",
            config.RAY_NUM_CPUS,
            len(ray.nodes()),
        )
    except Exception as exc:
        logger.exception("Failed to initialize Ray cluster.")
        raise RuntimeError("Ray cluster initialization failed.") from exc


@ray.remote
class DataStreamer:
    """Stateful actor that simulates concurrent camera telemetry streams."""

    def __init__(self) -> None:
        self._batch_counters: dict[str, int] = {
            camera_id: 0 for camera_id in config.CAMERA_IDS
        }
        self._rng = np.random.default_rng(config.RANDOM_SEED)

    def get_batch(self, camera_id: str) -> pd.DataFrame:
        """
        Produce the next batch for a single camera source.

        Args:
            camera_id: Identifier for the camera stream.

        Returns:
            DataFrame with telemetry columns and a 128-d embedding per frame.

        Raises:
            ValueError: If camera_id is not configured.
        """
        if camera_id not in self._batch_counters:
            raise ValueError(
                f"Unknown camera_id '{camera_id}'. "
                f"Expected one of {config.CAMERA_IDS}."
            )

        self._batch_counters[camera_id] += 1
        batch_number = self._batch_counters[camera_id]
        anomalous = self._is_anomalous(camera_id, batch_number)

        logger.debug(
            "Generating batch %d for %s (anomalous=%s).",
            batch_number,
            camera_id,
            anomalous,
        )

        return self._build_batch_dataframe(
            camera_id=camera_id,
            batch_number=batch_number,
            anomalous=anomalous,
        )

    def get_batch_metadata(self, camera_id: str) -> dict[str, Any]:
        """Return lightweight metadata for the most recently emitted batch."""
        if camera_id not in self._batch_counters:
            raise ValueError(f"Unknown camera_id '{camera_id}'.")

        batch_number = self._batch_counters[camera_id]
        return {
            "camera_id": camera_id,
            "batch_number": batch_number,
            "anomalous": self._is_anomalous(camera_id, batch_number),
        }

    @staticmethod
    def _is_anomalous(camera_id: str, batch_number: int) -> bool:
        """Determine whether the current batch should inject anomalies."""
        return (
            camera_id == config.ANOMALY_CAMERA_ID
            and batch_number >= config.ANOMALY_START_BATCH
        )

    def _build_batch_dataframe(
        self,
        camera_id: str,
        batch_number: int,
        anomalous: bool,
    ) -> pd.DataFrame:
        """Construct a batch of mock frame telemetry records."""
        timestamps = [
            datetime.now(timezone.utc).isoformat()
            for _ in range(config.FRAMES_PER_BATCH)
        ]
        frame_ids = [
            f"{camera_id}_b{batch_number:03d}_f{frame_idx:02d}"
            for frame_idx in range(config.FRAMES_PER_BATCH)
        ]

        if anomalous:
            brightness = self._rng.normal(
                config.FAILED_BRIGHTNESS_MEAN,
                config.FAILED_BRIGHTNESS_STD,
                config.FRAMES_PER_BATCH,
            )
            embeddings = self._rng.normal(
                config.EMBEDDING_DRIFT_MEAN,
                config.EMBEDDING_DRIFT_STD,
                (config.FRAMES_PER_BATCH, config.EMBEDDING_DIM),
            )
        else:
            brightness = self._rng.normal(
                config.BASELINE_BRIGHTNESS_MEAN,
                config.BASELINE_BRIGHTNESS_STD,
                config.FRAMES_PER_BATCH,
            )
            embeddings = self._rng.normal(
                config.EMBEDDING_MEAN,
                config.EMBEDDING_STD,
                (config.FRAMES_PER_BATCH, config.EMBEDDING_DIM),
            )

        blur = self._rng.normal(
            config.BASELINE_BLUR_MEAN,
            config.BASELINE_BLUR_STD,
            config.FRAMES_PER_BATCH,
        )

        brightness = np.clip(brightness, 0.0, 1.0)
        blur = np.clip(blur, 0.0, 1.0)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "camera_id": camera_id,
                "frame_id": frame_ids,
                "brightness_avg": brightness.astype(np.float64),
                "blur_metric": blur.astype(np.float64),
                "embedding": [row for row in embeddings],
            }
        )


def create_data_streamer() -> ray.actor.ActorHandle:
    """Instantiate the DataStreamer actor on the Ray cluster."""
    try:
        streamer = DataStreamer.remote()
        logger.info("DataStreamer actor created.")
        return streamer
    except Exception as exc:
        logger.exception("Failed to create DataStreamer actor.")
        raise RuntimeError("DataStreamer actor creation failed.") from exc


def fetch_batches_concurrently(
    streamer: ray.actor.ActorHandle,
    camera_ids: tuple[str, ...] | None = None,
) -> list[pd.DataFrame]:
    """
    Request the next batch from each camera concurrently via ray.get().

    Args:
        streamer: Handle to the DataStreamer actor.
        camera_ids: Optional subset of cameras; defaults to all configured IDs.

    Returns:
        List of DataFrames in the same order as camera_ids.
    """
    ids = camera_ids or config.CAMERA_IDS
    try:
        futures = [streamer.get_batch.remote(camera_id) for camera_id in ids]
        return ray.get(futures)
    except Exception as exc:
        logger.exception("Concurrent batch fetch failed.")
        raise RuntimeError("Failed to fetch camera batches.") from exc
