from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJ_ROOT / "models"

SEED = 42

TARGET = ["health"]

CATEGORICAL_COLUMNS = [
    "curb_loc",
    "spc_common",
    "guards",
    "sidewalk",
    "user_type",
    "borough",
]
NUMERICAL_COLUMNS = [
    "block_id",
    "tree_dbh",
    "steward",
    "postcode",
    "cncldist",
    "st_senate",
    "nta",
    "ct",
    "x_sp",
    "y_sp",
    "problems",
    "root_problems",
    "trunk_problems",
    "brch_problems",
]
FEATURE_COLUMNS = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS

BATCH_SIZE = 256
DROPOUT = 0.3
EPOCHS = 12
OPTIMIZER_LEARNING_RATE = 1e-4
SCHEDULER_STEP_SIZE = 3
SCHEDULER_GAMMA = 0.2

POSTCODE_OUTLIERS = 83
TREE_DBH_OUTLIERS = 50
