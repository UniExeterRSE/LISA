import os
from pathlib import Path

import dotenv
from loguru import logger

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJ_ROOT / "models"

# Original project data directories; likely not relevant for future use but kept for reproducibility
ONEDRIVE_DIR = os.environ.get("ONEDRIVE_DIR", "OneDrive")
PILOT_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "LISA 1 Pilot Data/Pilot 1/120724"
LABELLED_TEST_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "LISA 1 Pilot Data/P1_1608/160824_IMU_DC"
MAIN_DATA_DIR = RAW_DATA_DIR / ONEDRIVE_DIR / "Main Data Collection"

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
