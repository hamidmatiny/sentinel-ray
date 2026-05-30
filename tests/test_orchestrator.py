"""Unit tests for the PipelineOrchestrator incident protocol."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import pytest

import config
from orchestrator import (
    QABatchRecord,
    compute_rolling_quarantine_rate,
    count_drift_features,
    evaluate_circuit_breaker,
    handle_circuit_breaker_trip,
    write_incident_alert,
)


class TestQuarantineRate:
    def test_zero_when_no_rows_processed(self) -> None:
        rate = compute_rolling_quarantine_rate(deque(), window_batches=5)
        assert rate == 0.0

    def test_calculates_rate_across_recent_batches(self) -> None:
        history = deque(
            [
                QABatchRecord(batch_round=1, valid_count=4, quarantined_count=0),
                QABatchRecord(batch_round=2, valid_count=0, quarantined_count=4),
            ]
        )
        rate = compute_rolling_quarantine_rate(history, window_batches=5)
        assert rate == 0.5


class TestEvaluateCircuitBreaker:
    def test_trips_on_high_quarantine_rate(self) -> None:
        should_trip, reasons = evaluate_circuit_breaker(
            camera_id="camera_2",
            quarantine_rate=0.50,
            drift_report=None,
        )
        assert should_trip is True
        assert any(reason["type"] == "qa_quarantine_rate_exceeded" for reason in reasons)

    def test_does_not_trip_on_healthy_quarantine_rate(self) -> None:
        should_trip, reasons = evaluate_circuit_breaker(
            camera_id="camera_1",
            quarantine_rate=0.0,
            drift_report=None,
        )
        assert should_trip is False
        assert reasons == []

    def test_trips_on_multi_feature_drift(self) -> None:
        drift_report = {
            "status": "analyzed",
            "drift_detected": True,
            "features": {
                "brightness_avg": {"drift_detected": True, "p_value": 0.001},
                "blur_metric": {"drift_detected": True, "p_value": 0.002},
                "embedding": {"drift_detected": False, "p_value": 0.20},
            },
        }
        should_trip, reasons = evaluate_circuit_breaker(
            camera_id="camera_1",
            quarantine_rate=0.0,
            drift_report=drift_report,
        )
        assert should_trip is True
        assert any(reason["type"] == "multi_feature_drift" for reason in reasons)

    def test_count_drift_features(self) -> None:
        drift_report = {
            "status": "analyzed",
            "drift_detected": True,
            "features": {
                "brightness_avg": {"drift_detected": True},
                "blur_metric": {"drift_detected": True},
            },
        }
        assert count_drift_features(drift_report) == ["brightness_avg", "blur_metric"]


class TestIncidentProtocol:
    def test_writes_structured_alert_file(self, tmp_path: Path) -> None:
        reasons = [
            {
                "type": "qa_quarantine_rate_exceeded",
                "message": "QA quarantine rate exceeded.",
                "observed_rate": 1.0,
                "threshold": config.QA_QUARANTINE_RATE_THRESHOLD,
            }
        ]
        incident = handle_circuit_breaker_trip(
            camera_id="camera_2",
            batch_round=6,
            reasons=reasons,
            quarantine_rate=1.0,
            drift_report=None,
            alerts_dir=tmp_path,
        )

        assert Path(incident["alert_path"]).exists()
        assert incident["retraining"]["status"] == "queued"
        assert incident["retraining"]["request_id"]

    def test_alert_payload_contains_notification_channels(self, tmp_path: Path) -> None:
        payload = {
            "incident_id": "inc_test",
            "camera_id": "camera_2",
            "batch_round": 6,
            "severity": config.INCIDENT_SEVERITY,
            "reasons": [],
            "summary": "test",
            "metrics": {},
            "drift_report": None,
            "notification_channels": {
                "slack": {"channel": "#sentinel-ray-alerts"},
                "pagerduty": {"routing_key": "sentinel-ray-production"},
            },
        }
        alert_path = write_incident_alert(payload, alerts_dir=tmp_path)
        content = Path(alert_path).read_text(encoding="utf-8")
        assert "slack" in content
        assert "pagerduty" in content


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialize Ray for orchestrator actor integration tests."""
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2, logging_level="ERROR")
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestPipelineOrchestratorActor:
    def test_trips_circuit_breaker_for_failing_camera(self, ray_cluster, tmp_path: Path) -> None:
        import ray
        from orchestrator import PipelineOrchestrator

        orchestrator = PipelineOrchestrator.remote(alerts_dir=str(tmp_path))
        tripped_at: int | None = None
        for batch_round in range(1, 8):
            if batch_round < config.ANOMALY_START_BATCH:
                qa_results = [
                    {
                        "camera_id": "camera_2",
                        "status": "passed",
                        "valid_count": 4,
                        "quarantined_count": 0,
                    }
                ]
            else:
                qa_results = [
                    {
                        "camera_id": "camera_2",
                        "status": "failed",
                        "valid_count": 0,
                        "quarantined_count": 4,
                    }
                ]

            result = ray.get(
                orchestrator.process_batch_metrics.remote(
                    batch_round,
                    qa_results,
                    [],
                )
            )
            if result["incident_count"] > 0:
                tripped_at = batch_round
            if batch_round < config.ANOMALY_START_BATCH:
                assert result["incident_count"] == 0

        assert tripped_at == config.ANOMALY_START_BATCH
        summary = ray.get(orchestrator.get_incident_summary.remote())
        assert summary["total_incidents"] == 1
        assert "camera_2" in summary["tripped_cameras"]
