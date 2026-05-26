"""Configuration for Phase 1: Ray-based data ingestion stream."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Stream topology
# ---------------------------------------------------------------------------
CAMERA_IDS: tuple[str, ...] = ("camera_1", "camera_2", "camera_3")
NUM_CAMERAS: int = len(CAMERA_IDS)

# ---------------------------------------------------------------------------
# Mock stream sizing
# ---------------------------------------------------------------------------
TOTAL_BATCHES: int = 10
FRAMES_PER_BATCH: int = 4
EMBEDDING_DIM: int = 128

# Baseline signal characteristics (healthy camera)
BASELINE_BRIGHTNESS_MEAN: float = 0.65
BASELINE_BRIGHTNESS_STD: float = 0.05
BASELINE_BLUR_MEAN: float = 0.12
BASELINE_BLUR_STD: float = 0.03
EMBEDDING_MEAN: float = 0.0
EMBEDDING_STD: float = 1.0

# ---------------------------------------------------------------------------
# Anomaly injection (camera_2 hardware failure after batch 5)
# ---------------------------------------------------------------------------
ANOMALY_CAMERA_ID: str = "camera_2"
ANOMALY_START_BATCH: int = 6  # batches 1–5 are healthy; 6+ are anomalous
FAILED_BRIGHTNESS_MEAN: float = 0.02
FAILED_BRIGHTNESS_STD: float = 0.01
EMBEDDING_DRIFT_MEAN: float = 4.0
EMBEDDING_DRIFT_STD: float = 1.5

# ---------------------------------------------------------------------------
# QA / drift thresholds (used in later phases)
# ---------------------------------------------------------------------------
BRIGHTNESS_MIN_THRESHOLD: float = 0.15
BRIGHTNESS_MAX_THRESHOLD: float = 0.95
BLUR_MAX_THRESHOLD: float = 0.45
EMBEDDING_DRIFT_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# Ray runtime
# ---------------------------------------------------------------------------
RAY_NUM_CPUS: int = 4
RANDOM_SEED: int = 42
