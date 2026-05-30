# Sentinel Ray — Engineering Phase Report

This document records the design, implementation, and verification history of the **Automated Data Drift & QA Gatekeeper** pipeline. Each phase builds on the previous one to form a production-oriented MLOps data quality system.

---

## Phase 1 — Distributed Ingestion Engine (Ray Core)

**Status:** Complete  
**Primary module:** `ingestion_engine.py`

### Objective

Build a high-performance, distributed mock ingestion stream that simulates three concurrent camera telemetry feeds using Ray Core, establishing the foundation for downstream QA and drift analysis.

### Architecture

- **Ray cluster:** Local cluster initialized via `ray.init(ignore_reinit_error=True)` with configurable CPU allocation.
- **DataStreamer actor:** Stateful `@ray.remote` actor maintaining per-camera batch counters.
- **Concurrent fetch pattern:** Each batch round dispatches `get_batch.remote()` calls for all three cameras and resolves them with `ray.get()` for parallel retrieval.
- **Telemetry schema:** Each frame carries `timestamp`, `camera_id`, `frame_id`, `brightness_avg`, `blur_metric`, and a 128-dimensional NumPy embedding vector.

### Anomaly Injection

Starting at batch 6, `camera_2` simulates a hardware failure:

- Brightness collapses toward ~5.0 (0–255 scale)
- Embedding distribution shifts to mean ≈ 4.0 with elevated variance

This provides a controllable signal for later QA and drift testing.

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Concurrent multi-camera streaming | Ray actor with per-camera state + parallel `ray.get()` fan-out |
| Reproducible mock data | Centralized `RANDOM_SEED` in `config.py` |
| Structured observability | Python `logging` module with consistent format strings |

### Results

- 10 batch rounds × 3 cameras processed concurrently on a local Ray cluster
- Healthy cameras (`camera_1`, `camera_3`) maintain stable brightness (~165) and embedding statistics
- Anomalous `camera_2` batches show visible brightness and embedding shifts from batch 6 onward

---

## Phase 2 — Distributed QA Gatekeeper (Pandera)

**Status:** Complete  
**Primary module:** `qa_validator.py`

### Objective

Add a distributed schema validation layer that inspects every incoming batch **before** downstream processing, quarantining invalid rows without crashing the Ray runtime.

### Architecture

- **QAValidationWorker pool:** Three Ray actors (one per camera) running Pandera `DataFrameSchema` checks in parallel.
- **Validation rules:**
  - `brightness_avg` ∈ [10.0, 255.0]
  - `blur_metric` ≥ 0
  - `camera_id` matches `^camera_\d+$`
  - `embedding` is exactly 128 dimensions
- **Row-level split:** Invalid rows are isolated via Pandera lazy validation failure cases; valid rows continue downstream.
- **Quarantine routing:** Failed rows written to `data/quarantine/` as JSON payloads with companion `_reason.json` metadata files.

### Pipeline Integration

```
DataStreamer → QAValidationWorker pool → valid rows → telemetry
                              ↓
                     data/quarantine/
```

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Pandera built-in checks incompatible with Python 3.14 / pandas Series dispatch | Replaced with element-wise custom check functions (`_brightness_valid`, `_camera_id_valid`, etc.) |
| pyarrow unavailable on Python 3.14 | Switched quarantine storage from Parquet to JSON |
| pytest module import failures | Added `pytest.ini` with `pythonpath = .` for project-root imports |
| Ray actor logging visibility | Alerts emitted at WARNING level with structured rule failure details |
| Validation must not crash pipeline | All schema errors caught; unexpected exceptions return error status dict |

### Results

- **16/16** pytest cases passing for QA validation
- `camera_2` rows quarantined from batch 6 onward (20 total rows in a full 10-round run)
- `camera_1` and `camera_3` pass QA consistently throughout the run
- Pipeline completes cleanly with `total_quarantined_rows=20`

---

## Phase 3 — Distributed Data Drift Engine (SciPy + Ray)

**Status:** Complete  
**Primary module:** `drift_detector.py`

### Objective

Analyze QA-validated streaming data for **statistical drift** against a golden baseline reference dataset, using a stateful Ray actor with per-camera sliding windows and non-blocking remote analysis.

### Architecture

- **Golden baseline:** `generate_baseline_data(num_samples=1000)` produces a static reference DataFrame with:
  - `brightness_avg` ~ N(165, 10)
  - `blur_metric` ~ N(0.12, 0.03)
  - 128-D embeddings ~ N(0, 1)
- **DriftAnalyzer actor:** Loads baseline on initialization; maintains a per-camera buffer of QA-passed DataFrames.
- **Sliding window trigger:** When a camera buffer reaches 20 rows, the most recent 20 samples are compared against the baseline.
- **Statistical tests:**
  - **Numerical drift:** Two-sample Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`) on `brightness_avg` and `blur_metric`
  - **Embedding drift:** KS test on embedding L2 norms plus centroid cosine similarity and Euclidean distance for observability
- **Drift report:** Structured JSON dictionary with per-feature metrics and a top-level `drift_detected` boolean (α = 0.05 for KS p-values; configurable thresholds for embedding distance/similarity).

### Pipeline Integration

```
DataStreamer → QAValidationWorker → valid rows → DriftAnalyzer (async .remote())
                              ↓                           ↓
                     data/quarantine/              drift reports / alerts
```

In `main.py`, drift analysis is submitted via `accumulate_and_analyze.remote()` immediately after QA validation. Telemetry logging proceeds while Ray workers compute drift statistics; reports are collected at the end of each batch round via deferred `ray.get()`.

### Non-Blocking Design

Drift futures are fired before telemetry logging within each batch round, allowing the main loop to continue processing while statistical checks run on Ray workers. Results are resolved and logged after the round's telemetry output, minimizing stream back-pressure.

### Empty Input Handling

When a camera produces zero QA-valid rows (e.g., `camera_2` after batch 6), `DriftAnalyzer` returns a `skipped` status with `reason: empty_input` rather than raising an exception. Buffering status is returned when the window has not yet reached 20 samples.

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Separating schema failure from statistical drift | QA gate runs first; only `valid_df` rows reach `DriftAnalyzer` |
| Window management across variable batch sizes | Per-camera list buffer concatenated into sliding tail of 20 rows |
| Testability of Ray actor logic | Extracted pure `compute_drift_report()` function for unit tests independent of Ray |
| Embedding drift in high-dimensional space | Centroid-based cosine similarity + Euclidean distance vs golden baseline |
| Graceful degradation on empty batches | Early return with structured skip dict; no buffer mutation |

### Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `DRIFT_WINDOW_SIZE` | `20` | Sliding window sample count |
| `DRIFT_ALPHA` | `0.05` | KS test significance threshold |
| `DRIFT_BASELINE_SAMPLES` | `1000` | Golden reference dataset size |
| `GOLDEN_BASELINE_BRIGHTNESS_MEAN` | `165.0` | Baseline brightness center |
| `GOLDEN_BASELINE_BRIGHTNESS_STD` | `10.0` | Baseline brightness spread |
| `EMBEDDING_COSINE_SIM_THRESHOLD` | `0.85` | Report-only centroid similarity reference |
| `EMBEDDING_EUCLIDEAN_THRESHOLD` | `2.0` | Report-only centroid distance reference |

### Results

- **11/11** new drift pytest cases passing (baseline distribution, KS detection, embedding shift, empty input, Ray actor integration)
- Healthy cameras accumulate buffers and produce `buffering` → `analyzed` reports without false-positive drift under nominal conditions
- Intentionally shifted brightness windows produce `drift_detected: true` with p-values below α
- Pipeline logs high-visibility warnings: `⚠️ STATISTICAL DRIFT DETECTED | camera=... | feature=... | p-value=...`

---

## Phase 4 — Automated Orchestration & Self-Healing Loop

**Status:** Complete  
**Primary module:** `orchestrator.py`

### Objective

Close the loop between detection and action by orchestrating automated incident response when QA quarantine rates or multi-feature drift exceed policy thresholds. Containerize the full SentinelRay framework for reproducible local and production-adjacent deployment.

### Architecture — Self-Healing Data Loop

```
QA metrics + Drift reports
         ↓
PipelineOrchestrator (Ray Actor)
         ↓
  Circuit Breaker Evaluation
    ├── QA quarantine rate > 15% (rolling 5-batch window)
    └── Drift detected on ≥ 2 features
         ↓
  Incident Protocol (once per camera)
    ├── Write JSON alert → alerts/  (Slack + PagerDuty payload structure)
    └── Async POST → retraining pipeline webhook (Airflow DAG trigger simulation)
```

The orchestrator consumes **serialized QA metrics** (valid/quarantined counts) and **drift report dictionaries** after each batch round. It maintains per-camera rolling state and avoids duplicate incidents once a camera's circuit breaker is open.

### Circuit Breaker Rules

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Rolling QA quarantine rate | > 15% over last 5 batches | Trip breaker |
| Multi-feature statistical drift | ≥ 2 features with `drift_detected: true` | Trip breaker |

When tripped, the orchestrator:

1. **Alerting:** Writes a structured JSON incident file to `alerts/` containing severity, camera ID, timestamp, reason codes, Slack message payload, and PagerDuty event payload.
2. **Self-Healing:** Queues an asynchronous HTTP POST to `RETRAINING_WEBHOOK_URL` simulating an Airflow DAG trigger (`sentinel_ray_retrain_and_verify`). If the endpoint is unreachable, the request payload is persisted locally for audit.

### Pipeline Integration

In `main.py`, after drift reports are collected each batch round:

```python
orchestrator.process_batch_metrics.remote(batch_round, qa_metrics, drift_reports)
```

At shutdown, `get_incident_summary()` returns all tripped cameras, incident IDs, alert paths, and retraining webhook request IDs for explicit logging.

### Containerization

| File | Purpose |
|------|---------|
| `Dockerfile` | Python 3.12-slim, non-root `sentinel` user (UID 10001), dependency install, workspace setup |
| `docker-compose.yml` | Service definition with volume mounts (`data/quarantine/`, `alerts/`, `logs/`), 2 CPU / 2 GB limits, port 8265 |
| `.dockerignore` | Excludes venv, caches, and local artifacts from build context |

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Duplicate alerts on every failing batch | Per-camera `tripped` flag suppresses repeat incidents while breaker remains open |
| Ray actor receiving large DataFrames | Strip DataFrames before orchestrator; pass count metrics only |
| Webhook unavailable in local/dev | Graceful fallback: persist retraining payload to `alerts/` with `simulated_local_only` status |
| Resource thrashing in Docker | `docker-compose.yml` CPU/memory limits cap Ray worker footprint |
| Test isolation for orchestrator | Pure functions (`evaluate_circuit_breaker`, `compute_rolling_quarantine_rate`) tested without Ray |

### Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `ALERTS_DIR` | `alerts/` | Incident JSON output directory |
| `QA_QUARANTINE_RATE_THRESHOLD` | `0.15` | Rolling quarantine rate trip threshold |
| `ORCHESTRATOR_QA_WINDOW_BATCHES` | `5` | Batches included in quarantine rate window |
| `DRIFT_MIN_FEATURES_FOR_INCIDENT` | `2` | Minimum drifted features to trip breaker |
| `RETRAINING_WEBHOOK_URL` | `http://localhost:8080/api/v1/retrain` | Retraining pipeline endpoint |

### Results

- **9/9** orchestrator pytest cases passing (quarantine rate math, circuit breaker rules, alert file creation, Ray actor integration)
- Full pipeline run trips circuit breaker for `camera_2` at batch 6 when quarantine rate exceeds 15%
- Incident JSON alerts written to `alerts/` with Slack and PagerDuty payload structures
- Retraining webhook POST simulated asynchronously; payload persisted when endpoint is unreachable
- `docker compose up --build` runs the complete pipeline with host-mounted quarantine and alert directories

---

## Test Summary

| Phase | Test module | Cases | Status |
|-------|-------------|-------|--------|
| 2 | `tests/test_qa_validator.py` | 16 | Passing |
| 3 | `tests/test_drift_detector.py` | 11 | Passing |
| 4 | `tests/test_orchestrator.py` | 9 | Passing |

Run the full suite:

```bash
pytest tests/ -v
```

Inside Docker:

```bash
docker compose run --rm sentinel-ray pytest tests/ -v
```

## Run the Full Pipeline

Local:

```bash
python3 main.py
```

Docker:

```bash
mkdir -p data/quarantine alerts logs
docker compose up --build
```

Expected flow: Ray init → stream → QA validation → drift analysis → orchestration → incident alerts + retraining triggers.
