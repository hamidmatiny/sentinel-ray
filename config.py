"""Configuration for the Sentinel Ray data ingestion and QA pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
QUARANTINE_DIR: Path = DATA_DIR / "quarantine"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Stream topology
# ---------------------------------------------------------------------------
CAMERA_IDS: tuple[str, ...] = ("camera_1", "camera_2", "camera_3")
NUM_CAMERAS: int = len(CAMERA_IDS)
CAMERA_ID_PATTERN: str = r"^camera_\d+$"

# ---------------------------------------------------------------------------
# Mock stream sizing
# ---------------------------------------------------------------------------
TOTAL_BATCHES: int = 10
FRAMES_PER_BATCH: int = 4
EMBEDDING_DIM: int = 128

# Baseline signal characteristics (healthy camera; brightness on 0–255 scale)
BASELINE_BRIGHTNESS_MEAN: float = 165.0
BASELINE_BRIGHTNESS_STD: float = 12.0
BASELINE_BLUR_MEAN: float = 0.12
BASELINE_BLUR_STD: float = 0.03
EMBEDDING_MEAN: float = 0.0
EMBEDDING_STD: float = 1.0

# ---------------------------------------------------------------------------
# Anomaly injection (camera_2 hardware failure after batch 5)
# ---------------------------------------------------------------------------
ANOMALY_CAMERA_ID: str = "camera_2"
ANOMALY_START_BATCH: int = 6  # batches 1–5 are healthy; 6+ are anomalous
FAILED_BRIGHTNESS_MEAN: float = 5.0
FAILED_BRIGHTNESS_STD: float = 2.0
EMBEDDING_DRIFT_MEAN: float = 4.0
EMBEDDING_DRIFT_STD: float = 1.5

# ---------------------------------------------------------------------------
# QA validation thresholds
# ---------------------------------------------------------------------------
BRIGHTNESS_MIN_VALID: float = 10.0
BRIGHTNESS_MAX_VALID: float = 255.0
BLUR_MIN_VALID: float = 0.0
EMBEDDING_DRIFT_THRESHOLD: float = 2.0

# Legacy drift thresholds (reserved for Phase 3)
BRIGHTNESS_MIN_THRESHOLD: float = BRIGHTNESS_MIN_VALID
BRIGHTNESS_MAX_THRESHOLD: float = BRIGHTNESS_MAX_VALID
BLUR_MAX_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Ray runtime
# ---------------------------------------------------------------------------
RAY_NUM_CPUS: int = 4
QA_WORKER_POOL_SIZE: int = NUM_CAMERAS
RANDOM_SEED: int = 42
