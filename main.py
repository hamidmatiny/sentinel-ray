"""Phase 1 entry point: distributed multi-camera ingestion stream."""

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


def configure_logging() -> None:
    """Configure structured console logging for the pipeline."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def summarize_batch(batch_df: pd.DataFrame) -> dict[str, Any]:
    """Compute compact telemetry summary for console output."""
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


def log_batch_telemetry(batch_index: int, summaries: list[dict[str, Any]]) -> None:
    """Emit structured stream telemetry for one concurrent fetch round."""
    logger = logging.getLogger(__name__)
    logger.info("=== Batch round %d / %d ===", batch_index, config.TOTAL_BATCHES)

    for summary in summaries:
        logger.info(
            (
                "camera=%s | frames=%d | brightness_mean=%.4f | "
                "brightness_min=%.4f | blur_mean=%.4f | "
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
    """Initialize Ray, consume batches concurrently, and log telemetry."""
    logger = logging.getLogger(__name__)
    streamer = None

    try:
        initialize_ray_cluster()
        streamer = create_data_streamer()

        logger.info(
            "Starting ingestion: %d batch rounds x %d cameras.",
            config.TOTAL_BATCHES,
            config.NUM_CAMERAS,
        )

        for batch_index in range(1, config.TOTAL_BATCHES + 1):
            batch_frames = fetch_batches_concurrently(streamer)
            summaries = [summarize_batch(df) for df in batch_frames]
            log_batch_telemetry(batch_index, summaries)

        logger.info("Ingestion pipeline completed successfully.")

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
