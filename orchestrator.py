"""Automated orchestration layer for QA and drift incident response."""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import ray

import config

logger = logging.getLogger(__name__)


@dataclass
class QABatchRecord:
    """QA throughput metrics captured for a single batch round."""

    batch_round: int
    valid_count: int
    quarantined_count: int

    @property
    def total_count(self) -> int:
        return self.valid_count + self.quarantined_count

    @property
    def quarantine_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.quarantined_count / self.total_count


@dataclass
class CameraHealthState:
    """Rolling health state tracked per camera."""

    qa_history: deque[QABatchRecord] = field(default_factory=deque)
    tripped: bool = False
    incident_ids: list[str] = field(default_factory=list)


def compute_rolling_quarantine_rate(
    qa_history: deque[QABatchRecord],
    window_batches: int,
) -> float:
    """Calculate quarantine rate across the most recent batch window."""
    if not qa_history:
        return 0.0

    recent_records = list(qa_history)[-window_batches:]
    total_rows = sum(record.total_count for record in recent_records)
    if total_rows == 0:
        return 0.0

    quarantined_rows = sum(record.quarantined_count for record in recent_records)
    return quarantined_rows / total_rows


def count_drift_features(drift_report: dict[str, Any]) -> list[str]:
    """Return feature names that flagged drift in an analyzed report."""
    if drift_report.get("status") != "analyzed" or not drift_report.get("drift_detected"):
        return []

    drifted_features: list[str] = []
    for feature_name, metrics in drift_report.get("features", {}).items():
        if metrics.get("drift_detected"):
            drifted_features.append(str(feature_name))
    return drifted_features


def evaluate_circuit_breaker(
    camera_id: str,
    quarantine_rate: float,
    drift_report: dict[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    """
    Evaluate whether a camera should trip the system circuit breaker.

    Returns:
        Tuple of (should_trip, list of structured reason dicts).
    """
    reasons: list[dict[str, Any]] = []

    if quarantine_rate > config.QA_QUARANTINE_RATE_THRESHOLD:
        reasons.append(
            {
                "type": "qa_quarantine_rate_exceeded",
                "camera_id": camera_id,
                "observed_rate": round(quarantine_rate, 4),
                "threshold": config.QA_QUARANTINE_RATE_THRESHOLD,
                "message": (
                    f"QA quarantine rate {quarantine_rate:.2%} exceeds "
                    f"{config.QA_QUARANTINE_RATE_THRESHOLD:.2%} threshold."
                ),
            }
        )

    if drift_report is not None:
        drifted_features = count_drift_features(drift_report)
        if len(drifted_features) >= config.DRIFT_MIN_FEATURES_FOR_INCIDENT:
            reasons.append(
                {
                    "type": "multi_feature_drift",
                    "camera_id": camera_id,
                    "features": drifted_features,
                    "feature_count": len(drifted_features),
                    "threshold": config.DRIFT_MIN_FEATURES_FOR_INCIDENT,
                    "message": (
                        f"Drift detected across {len(drifted_features)} features: "
                        f"{', '.join(drifted_features)}."
                    ),
                }
            )

    return bool(reasons), reasons


def build_incident_payload(
    camera_id: str,
    batch_round: int,
    reasons: list[dict[str, Any]],
    quarantine_rate: float,
    drift_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a structured incident alert payload for webhook delivery."""
    incident_id = f"inc_{camera_id}_{uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    drift_summary: dict[str, Any] | None = None
    if drift_report and drift_report.get("status") == "analyzed":
        drift_summary = {
            "drift_detected": drift_report.get("drift_detected", False),
            "window_size": drift_report.get("window_size", 0),
            "features": drift_report.get("features", {}),
        }

    return {
        "incident_id": incident_id,
        "severity": config.INCIDENT_SEVERITY,
        "status": "open",
        "timestamp": timestamp,
        "batch_round": batch_round,
        "camera_id": camera_id,
        "circuit_breaker_tripped": True,
        "summary": (
            f"Circuit breaker tripped for {camera_id}: "
            + "; ".join(reason["message"] for reason in reasons)
        ),
        "reasons": reasons,
        "metrics": {
            "rolling_quarantine_rate": round(quarantine_rate, 4),
            "qa_window_batches": config.ORCHESTRATOR_QA_WINDOW_BATCHES,
        },
        "drift_report": drift_summary,
        "notification_channels": {
            "slack": {
                "channel": "#sentinel-ray-alerts",
                "text": (
                    f":rotating_light: *SentinelRay Incident* | camera `{camera_id}` | "
                    f"batch {batch_round} | quarantine_rate={quarantine_rate:.2%}"
                ),
            },
            "pagerduty": {
                "routing_key": "sentinel-ray-production",
                "event_action": "trigger",
                "payload": {
                    "summary": f"SentinelRay circuit breaker tripped for {camera_id}",
                    "severity": config.INCIDENT_SEVERITY,
                    "source": "sentinel-ray-orchestrator",
                    "custom_details": {
                        "camera_id": camera_id,
                        "batch_round": batch_round,
                        "reasons": reasons,
                    },
                },
            },
        },
    }


def write_incident_alert(incident_payload: dict[str, Any], alerts_dir: Path | None = None) -> str:
    """Persist an incident alert JSON payload to the alerts directory."""
    target_dir = alerts_dir or config.ALERTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    incident_id = incident_payload["incident_id"]
    camera_id = incident_payload["camera_id"]
    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    alert_path = target_dir / f"{incident_id}_{camera_id}_{timestamp_slug}.json"
    alert_path.write_text(json.dumps(incident_payload, indent=2), encoding="utf-8")

    logger.warning(
        "INCIDENT ALERT WRITTEN | incident_id=%s | camera=%s | path=%s",
        incident_id,
        camera_id,
        alert_path,
    )
    return str(alert_path)


def build_retraining_payload(
    incident_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build the payload for a simulated retraining pipeline trigger."""
    return {
        "request_id": f"retrain_{uuid4().hex[:12]}",
        "trigger_source": "sentinel-ray-orchestrator",
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "incident_id": incident_payload["incident_id"],
        "camera_id": incident_payload["camera_id"],
        "pipeline": "upstream-model-retrain",
        "engine": "airflow",
        "action": "trigger_dag",
        "dag_id": "sentinel_ray_retrain_and_verify",
        "inputs": {
            "quarantine_dir": str(config.QUARANTINE_DIR),
            "alerts_dir": str(config.ALERTS_DIR),
            "reasons": incident_payload["reasons"],
        },
        "metadata": {
            "batch_round": incident_payload["batch_round"],
            "severity": incident_payload["severity"],
        },
    }


def _execute_retraining_request(
    retraining_payload: dict[str, Any],
    alerts_dir: Path,
) -> dict[str, Any]:
    """Simulate an asynchronous HTTP POST to a retraining pipeline engine."""
    alerts_dir.mkdir(parents=True, exist_ok=True)
    request_path = alerts_dir / f"{retraining_payload['request_id']}_retraining.json"
    request_path.write_text(json.dumps(retraining_payload, indent=2), encoding="utf-8")

    response_record: dict[str, Any] = {
        "request_id": retraining_payload["request_id"],
        "request_path": str(request_path),
        "webhook_url": config.RETRAINING_WEBHOOK_URL,
        "status": "simulated",
        "http_status": None,
        "error": None,
    }

    request_body = json.dumps(retraining_payload).encode("utf-8")
    http_request = urllib.request.Request(
        config.RETRAINING_WEBHOOK_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            http_request,
            timeout=config.RETRAINING_REQUEST_TIMEOUT_SEC,
        ) as http_response:
            response_record["status"] = "submitted"
            response_record["http_status"] = http_response.status
            logger.info(
                "RETRAINING WEBHOOK POST | request_id=%s | status=%s",
                retraining_payload["request_id"],
                http_response.status,
            )
    except urllib.error.URLError as exc:
        response_record["status"] = "simulated_local_only"
        response_record["error"] = str(exc.reason)
        logger.info(
            (
                "RETRAINING WEBHOOK SIMULATED | request_id=%s | "
                "payload_saved=%s | webhook_unreachable=%s"
            ),
            retraining_payload["request_id"],
            request_path,
            exc.reason,
        )
    except Exception as exc:
        response_record["status"] = "failed"
        response_record["error"] = str(exc)
        logger.exception(
            "Unexpected retraining webhook failure for request_id=%s.",
            retraining_payload["request_id"],
        )

    return response_record


def trigger_retraining_async(
    incident_payload: dict[str, Any],
    alerts_dir: Path | None = None,
) -> dict[str, Any]:
    """Fire a background retraining trigger and return immediate metadata."""
    target_dir = alerts_dir or config.ALERTS_DIR
    retraining_payload = build_retraining_payload(incident_payload)

    thread = threading.Thread(
        target=_execute_retraining_request,
        args=(retraining_payload, target_dir),
        daemon=True,
        name=f"retrain-{incident_payload['camera_id']}",
    )
    thread.start()

    return {
        "request_id": retraining_payload["request_id"],
        "status": "queued",
        "webhook_url": config.RETRAINING_WEBHOOK_URL,
    }


def handle_circuit_breaker_trip(
    camera_id: str,
    batch_round: int,
    reasons: list[dict[str, Any]],
    quarantine_rate: float,
    drift_report: dict[str, Any] | None,
    alerts_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute incident protocol actions when the circuit breaker trips."""
    incident_payload = build_incident_payload(
        camera_id=camera_id,
        batch_round=batch_round,
        reasons=reasons,
        quarantine_rate=quarantine_rate,
        drift_report=drift_report,
    )
    alert_path = write_incident_alert(incident_payload, alerts_dir=alerts_dir)
    retraining_result = trigger_retraining_async(incident_payload, alerts_dir=alerts_dir)

    return {
        "incident_id": incident_payload["incident_id"],
        "camera_id": camera_id,
        "batch_round": batch_round,
        "alert_path": alert_path,
        "retraining": retraining_result,
        "reasons": reasons,
    }


def create_pipeline_orchestrator() -> ray.actor.ActorHandle:
    """Instantiate the PipelineOrchestrator actor on the Ray cluster."""
    try:
        orchestrator = PipelineOrchestrator.remote()
        logger.info("PipelineOrchestrator actor created.")
        return orchestrator
    except Exception as exc:
        logger.exception("Failed to create PipelineOrchestrator actor.")
        raise RuntimeError("PipelineOrchestrator actor creation failed.") from exc


@ray.remote
class PipelineOrchestrator:
    """Tracks system health and executes incident response for QA and drift failures."""

    def __init__(self, alerts_dir: str | None = None) -> None:
        self._alerts_dir = Path(alerts_dir or config.ALERTS_DIR)
        self._alerts_dir.mkdir(parents=True, exist_ok=True)
        self._camera_states: dict[str, CameraHealthState] = {
            camera_id: CameraHealthState() for camera_id in config.CAMERA_IDS
        }
        self._incidents: list[dict[str, Any]] = []
        self._webhook_triggers: list[dict[str, Any]] = []

    def process_batch_metrics(
        self,
        batch_round: int,
        qa_results: list[dict[str, Any]],
        drift_reports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Consume QA and drift outputs for a batch round and evaluate circuit breakers.

        Returns:
            Structured dict describing any incidents triggered in this round.
        """
        drift_by_camera = {
            report.get("camera_id"): report
            for report in drift_reports
            if report.get("camera_id")
        }

        round_incidents: list[dict[str, Any]] = []

        for qa_result in qa_results:
            camera_id = str(qa_result.get("camera_id", "unknown"))
            if camera_id not in self._camera_states:
                logger.warning(
                    "Orchestrator ignoring unknown camera_id=%s in batch_round=%d.",
                    camera_id,
                    batch_round,
                )
                continue

            state = self._camera_states[camera_id]
            qa_record = QABatchRecord(
                batch_round=batch_round,
                valid_count=int(qa_result.get("valid_count", 0)),
                quarantined_count=int(qa_result.get("quarantined_count", 0)),
            )
            state.qa_history.append(qa_record)

            quarantine_rate = compute_rolling_quarantine_rate(
                state.qa_history,
                config.ORCHESTRATOR_QA_WINDOW_BATCHES,
            )
            drift_report = drift_by_camera.get(camera_id)
            should_trip, reasons = evaluate_circuit_breaker(
                camera_id=camera_id,
                quarantine_rate=quarantine_rate,
                drift_report=drift_report,
            )

            if not should_trip:
                continue

            if state.tripped:
                logger.info(
                    "CIRCUIT BREAKER OPEN | camera=%s | batch_round=%d | "
                    "skipping duplicate incident.",
                    camera_id,
                    batch_round,
                )
                continue

            try:
                incident = handle_circuit_breaker_trip(
                    camera_id=camera_id,
                    batch_round=batch_round,
                    reasons=reasons,
                    quarantine_rate=quarantine_rate,
                    drift_report=drift_report,
                    alerts_dir=self._alerts_dir,
                )
                state.tripped = True
                state.incident_ids.append(incident["incident_id"])
                self._incidents.append(incident)
                self._webhook_triggers.append(incident["retraining"])
                round_incidents.append(incident)

                logger.critical(
                    "CIRCUIT BREAKER TRIPPED | camera=%s | batch_round=%d | "
                    "incident_id=%s | quarantine_rate=%.2f%%",
                    camera_id,
                    batch_round,
                    incident["incident_id"],
                    quarantine_rate * 100,
                )
            except Exception:
                logger.exception(
                    "Failed to execute incident protocol for camera=%s batch_round=%d.",
                    camera_id,
                    batch_round,
                )

        return {
            "batch_round": batch_round,
            "incidents_triggered": round_incidents,
            "incident_count": len(round_incidents),
        }

    def get_incident_summary(self) -> dict[str, Any]:
        """Return aggregate incident and webhook statistics for pipeline shutdown."""
        tripped_cameras = [
            camera_id
            for camera_id, state in self._camera_states.items()
            if state.tripped
        ]
        return {
            "total_incidents": len(self._incidents),
            "tripped_cameras": tripped_cameras,
            "incidents": self._incidents,
            "webhook_triggers": self._webhook_triggers,
        }
