import os
from pathlib import Path

import dotenv
from loguru import logger

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

ONEDRIVE_DIR = os.environ.get("ONEDRIVE_DIR", "OneDrive")

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
PILOT_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "LISA 1 Pilot Data/Pilot 1/120724"
LABELLED_TEST_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "LISA 1 Pilot Data/P1_1608/160824_IMU_DC"
MAIN_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "Main Data Collection"

MODELS_DIR = PROJ_ROOT / "models"
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"

# Common Regex Patterns
IMU_PATTERN = r"^(.*?)_(.*?)_(.*?)\.(.*?)$"
FOOT_SENSOR_PATTERN = r"^(.*?)_(left foot sensor|right foot sensor)\..*$"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

HYPERPARAMETERS = {
    "LR": {},
    "RF": {
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 8,
        "min_samples_split": 8,
        "n_estimators": 28,
        "bootstrap": True,
    },
    "LGBM": {
        "bagging_fraction": 0.6,
        "bagging_freq": 5,
        "extra_trees": True,
        "feature_fraction": 0.6,
        "lambda_l1": 0.0,
        "lambda_l2": 0.1,
        "max_bin": 255,
        "max_depth": 3,
        "min_data_in_leaf": 50,
        "min_gain_to_split": 0.1,
        "min_sum_hessian_in_leaf": 0.1,
        "num_leaves": 63,
        "path_smooth": 0.3,
    },
}
