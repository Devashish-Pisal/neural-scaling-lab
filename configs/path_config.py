from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


# Folders
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"
WANDB_ARCHIVE_DIR = DATA_DIR / "wandb_archive"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PDF_OUTPUTS_DIR = OUTPUT_DIR / "pdfs"
IMG_OUTPUT_DIR = OUTPUT_DIR / "images"
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


# Files
MAIN_EXPERIMENT_RUNS_FILE_PATH = DATABASE_DIR / "main_experiment_runs.csv"
EXTREME_BG_RUNS_FILE_PATH = DATABASE_DIR / "extreme_bg_runs.csv"