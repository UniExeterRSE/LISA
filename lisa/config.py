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

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
