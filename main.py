"""Entry point: ingestion stream with QA validation, drift analysis, and orchestration."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
import ray

import config
from drift_detector import create_drift_analyzer
from ingestion_engine import (
    create_data_streamer,
    fetch_batches_concurrently,
    initialize_ray_cluster,
)
from orchestrator import create_pipeline_orchestrator
from qa_validator import create_qa_worker_pool, validate_batches_concurrently


def configure_logging() -> None:
    """Configure structured console logging for the pipeline."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    config.ALERTS_DIR.mkdir(parents=True, exist_ok=True)
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


def log_drift_report(report: dict[str, Any]) -> None:
    """Emit drift analysis results, with high-visibility warnings on detection."""
    logger = logging.getLogger(__name__)
    camera_id = report.get("camera_id", "unknown")
    status = report.get("status", "unknown")

    if status == "buffering":
        logger.info(
            "DRIFT BUFFER | camera=%s | buffer_size=%d | required=%d",
            camera_id,
            report.get("buffer_size", 0),
            report.get("window_size_required", config.DRIFT_WINDOW_SIZE),
        )
        return

    if status == "skipped":
        logger.debug(
            "DRIFT SKIPPED | camera=%s | reason=%s",
            camera_id,
            report.get("reason", "unknown"),
        )
        return

    if status != "analyzed":
        return

    logger.info(
        "DRIFT REPORT | camera=%s | window_size=%d | drift_detected=%s",
        camera_id,
        report.get("window_size", 0),
        report.get("drift_detected", False),
    )

    if not report.get("drift_detected"):
        return

    for feature, metrics in report.get("features", {}).items():
        if not metrics.get("drift_detected"):
            continue

        if "p_value" in metrics:
            logger.warning(
                "⚠️ STATISTICAL DRIFT DETECTED | camera=%s | feature=%s | p-value=%.6f",
                camera_id,
                feature,
                metrics["p_value"],
            )
        else:
            logger.warning(
                (
                    "⚠️ STATISTICAL DRIFT DETECTED | camera=%s | feature=%s | "
                    "cosine_sim=%.4f | euclidean_dist=%.4f"
                ),
                camera_id,
                feature,
                metrics.get("mean_cosine_similarity", 0.0),
                metrics.get("mean_euclidean_distance", 0.0),
            )


def log_orchestrator_incidents(batch_result: dict[str, Any]) -> None:
    """Log incidents triggered by the orchestrator for a batch round."""
    logger = logging.getLogger(__name__)
    for incident in batch_result.get("incidents_triggered", []):
        logger.critical(
            (
                "ORCHESTRATOR INCIDENT | batch_round=%d | camera=%s | "
                "incident_id=%s | alert_path=%s | retraining_request_id=%s"
            ),
            incident.get("batch_round"),
            incident.get("camera_id"),
            incident.get("incident_id"),
            incident.get("alert_path"),
            incident.get("retraining", {}).get("request_id"),
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


def submit_drift_analysis(
    drift_analyzer: ray.actor.ActorHandle,
    qa_results: list[dict[str, Any]],
) -> list[tuple[str, ray.ObjectRef]]:
    """Fire non-blocking drift analysis tasks for QA-validated camera batches."""
    futures: list[tuple[str, ray.ObjectRef]] = []
    for result in qa_results:
        valid_df = result["valid_df"]
        camera_id = result["camera_id"]
        future = drift_analyzer.accumulate_and_analyze.remote(camera_id, valid_df)
        futures.append((camera_id, future))
    return futures


def collect_drift_reports(
    drift_futures: list[tuple[str, ray.ObjectRef]],
) -> list[dict[str, Any]]:
    """Resolve drift analysis futures and log any detected drift."""
    reports: list[dict[str, Any]] = []
    for camera_id, future in drift_futures:
        try:
            report = ray.get(future)
            reports.append(report)
            log_drift_report(report)
        except Exception:
            logging.getLogger(__name__).exception(
                "Drift analysis failed for camera=%s.",
                camera_id,
            )
    return reports


def serialize_qa_results_for_orchestrator(
    qa_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip DataFrames from QA results before sending metrics to the orchestrator."""
    serialized: list[dict[str, Any]] = []
    for result in qa_results:
        serialized.append(
            {
                "camera_id": result.get("camera_id"),
                "status": result.get("status"),
                "valid_count": result.get("valid_count", 0),
                "quarantined_count": result.get("quarantined_count", 0),
                "error_message": result.get("error_message"),
            }
        )
    return serialized


def log_pipeline_summary(
    total_quarantined: int,
    drift_alerts: int,
    incident_summary: dict[str, Any],
) -> None:
    """Emit final pipeline statistics including orchestrator incidents."""
    logger = logging.getLogger(__name__)
    logger.info(
        (
            "Ingestion pipeline completed. total_quarantined_rows=%d | "
            "drift_alert_count=%d | total_incidents=%d"
        ),
        total_quarantined,
        drift_alerts,
        incident_summary.get("total_incidents", 0),
    )

    tripped_cameras = incident_summary.get("tripped_cameras", [])
    if tripped_cameras:
        logger.critical(
            "CIRCUIT BREAKER SUMMARY | tripped_cameras=%s",
            ", ".join(tripped_cameras),
        )

    for incident in incident_summary.get("incidents", []):
        logger.critical(
            (
                "INCIDENT SUMMARY | camera=%s | incident_id=%s | "
                "alert_path=%s | reasons=%d"
            ),
            incident.get("camera_id"),
            incident.get("incident_id"),
            incident.get("alert_path"),
            len(incident.get("reasons", [])),
        )

    for webhook in incident_summary.get("webhook_triggers", []):
        logger.info(
            (
                "WEBHOOK SUMMARY | request_id=%s | status=%s | url=%s"
            ),
            webhook.get("request_id"),
            webhook.get("status"),
            webhook.get("webhook_url"),
        )


def run_ingestion_pipeline() -> None:
    """Initialize Ray, validate batches, analyze drift, orchestrate incidents."""
    logger = logging.getLogger(__name__)
    streamer = None

    try:
        initialize_ray_cluster()
        streamer = create_data_streamer()
        qa_workers = create_qa_worker_pool()
        drift_analyzer = create_drift_analyzer()
        orchestrator = create_pipeline_orchestrator()

        logger.info(
            (
                "Starting SentinelRay pipeline: %d batch rounds x %d cameras "
                "(QA + drift + orchestration)."
            ),
            config.TOTAL_BATCHES,
            config.NUM_CAMERAS,
        )

        total_quarantined = 0
        drift_alerts = 0

        for batch_index in range(1, config.TOTAL_BATCHES + 1):
            raw_batches = fetch_batches_concurrently(streamer)
            qa_results = validate_batches_concurrently(
                qa_workers,
                raw_batches,
                batch_round=batch_index,
            )

            drift_futures = submit_drift_analysis(drift_analyzer, qa_results)

            valid_batches = [result["valid_df"] for result in qa_results]
            summaries = []
            for result, valid_df in zip(qa_results, valid_batches):
                summary = summarize_batch(valid_df)
                if summary["frame_count"] == 0:
                    summary["camera_id"] = result["camera_id"]
                summaries.append(summary)
            log_batch_telemetry(batch_index, summaries, qa_results)

            drift_reports = collect_drift_reports(drift_futures)
            drift_alerts += sum(1 for report in drift_reports if report.get("drift_detected"))

            orchestrator_input = serialize_qa_results_for_orchestrator(qa_results)
            batch_incidents = ray.get(
                orchestrator.process_batch_metrics.remote(
                    batch_index,
                    orchestrator_input,
                    drift_reports,
                )
            )
            log_orchestrator_incidents(batch_incidents)

            total_quarantined += sum(
                result["quarantined_count"] for result in qa_results
            )

        incident_summary = ray.get(orchestrator.get_incident_summary.remote())
        log_pipeline_summary(total_quarantined, drift_alerts, incident_summary)

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
