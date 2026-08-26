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
# Only 2 database files (2026-08-25 confirmed, re-derived from thesis
# Chapter 4): main_experiment_runs.csv now covers BOTH the Scaling Law
# Analysis (Table 4.4, all 8 models x 3 baselines) and the
# Foreground-Background Contribution Analysis (Table 4.5, 5 models x full
# grid x 300 epochs) - they share a schema where crossover epochs are
# meaningful. extreme_bg_runs.csv remains separate because crossover is
# NOT meaningful for the Background Subset Size Ablation (Table 4.6).
# There is no separate pareto-frontier file - that concept from the
# internal planning notes was patch-16-only and is superseded by the
# thesis's all-8-models Scaling Law Analysis.
MAIN_EXPERIMENT_RUNS_FILE_PATH = DATABASE_DIR / "main_experiment_runs.csv"
EXTREME_BG_RUNS_FILE_PATH = DATABASE_DIR / "extreme_bg_runs.csv"