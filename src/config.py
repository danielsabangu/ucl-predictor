from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODELS_DIR = BASE_DIR / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
PREPROCESSING_DIR = MODELS_DIR / "preprocessing"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"

REPORTS_DIR = BASE_DIR / "reports"

FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")