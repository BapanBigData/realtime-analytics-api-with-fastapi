from pathlib import Path
from datetime import datetime

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "features" / "training_set"
MODEL_DIR = BASE_DIR / "outputs" / "models"
LOG_DIR = BASE_DIR / "outputs" / "logs"
SCALER_STATS_DIR = BASE_DIR / "outputs" / "scaler_stats"


# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCALER_STATS_DIR.mkdir(parents=True, exist_ok=True)

# Files
TRAIN_DATA_PATH = DATA_DIR
MODEL_FILE = MODEL_DIR / "xgb_model.joblib"
LOG_FILE = LOG_DIR / f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Settings
RANDOM_SEED = 47
TEST_SIZE = 0.33
