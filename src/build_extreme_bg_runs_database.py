import os
import wandb
import pandas as pd
from wandb import Run
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from configs.config import EXTREME_BG_ID_MAPPING, EXTREME_BG_INSCOPE_MODELS, BASELINE_EPOCH_SCHEDULES
from configs.path_config import EXTREME_BG_RUNS_FILE_PATH
from src.build_main_experiment_runs_database import create_database, check_ds_fraction_and_run_id_mapping, is_fg_range_and_epochs_mapping_correct, contains_run_name_keywords, save_run_data_to_db


# Full valid bg_range grid for the Background Subset Size Ablation (thesis
# Table 4.6), as of the 2026-08-25 revision. Union of:
#  - the target grid (7 points, 0.0001%-100%): 0-0.0001, 0-0.001, 0-0.01,
#    0-0.1, 0-1, 0-10, 0-100
#  - the pre-existing ViT-S/16 oip=cos pilot grid (7 points, 0.0001%-5%,
#    left as-is): 0-0.0001, 0-0.001, 0-0.01, 0-0.05, 0-0.1, 0-1, 0-5
VALID_EXTREME_BG_RANGES = ("0-0.0001", "0-0.001", "0-0.01", "0-0.05", "0-0.1", "0-1", "0-5", "0-10", "0-100")

# ASSUMPTION (thesis Table 4.6 does not restate a baseline epoch count for
# this ablation): fixed at 300, matching the pre-existing pilot data and
# the fact fg_range is fixed at "0-100" throughout. Confirm before relying
# on this if the ablation is ever run at a different baseline.
EXTREME_BG_BASELINE_EPOCHS = 300


def validate_extreme_bg_fornet_run(run:Run, expected_model:str, bg_fraction:float, oip_prob:str, baseline_epochs:int = EXTREME_BG_BASELINE_EPOCHS):
    id = run.id
    name = run.name
    config = run.config
    expected_epochs = BASELINE_EPOCH_SCHEDULES[baseline_epochs]["0-100"]
    if run.state != "finished":
        raise ValueError(f"Run {id} | Run state is not 'finished'. Expected run state 'finished' but got '{run.state}' | Run name '{name}'")
    if config["model"].lower().replace("/", "-") != expected_model.lower().replace("/", "-"):
        raise ValueError(f"Run {id} | Model mismatch. Expected '{expected_model}' but got '{config['model']}'")
    if not "fornet" in str(name):
        raise ValueError(f"Run {id} | Run does not contains 'fornet' in its run name | Run name '{name}'")
    if not config["dataset"] in ("fornet/all/cos", "fornet/all/0.0"):
        raise ValueError(f"Run {id} | Dataset mismatch. Expected 'fornet/all/cos or fornet/all/0.0' but got {config["dataset"]} | Run name '{name}'")
    if config["epochs"] != expected_epochs:
        raise ValueError(f"Run {id} | Invalid Epochs | Extreme-BG runs are fixed at fg_range='0-100', baseline_epochs={baseline_epochs} -> {expected_epochs} epochs. Got {config["epochs"]} | Run name '{name}'")
    if config["dataset"].split("/")[2] != oip_prob.split("-")[1]:
        raise ValueError(f"Run {id} | Original image probability mismatch. Expected '{config["dataset"].split("/")[2]}' but got '{oip_prob.split("-")[1]}' | Run name '{name}'")
    if config["fg_range"] != "0-100":
        raise ValueError(f"Run {id} | Invalid FG range | All extreme-BG runs, for any model, must use fg_range='0-100' (100% foreground pool) per thesis Chapter 4 - the earlier fg_range='0-10' ViT-B/16 pilot is discarded and must not be re-added. Expected '0-100' but got {config["fg_range"]} | Run name '{name}'")
    if config["bg_range"] not in VALID_EXTREME_BG_RANGES:
        raise ValueError(f"Run {id} | Invalid BG range | Expected one of {VALID_EXTREME_BG_RANGES} but got {config["bg_range"]} | Run name '{name}'")
    range = config["bg_range"].split("-")
    actual_bg_fraction = (float(range[1]) - float(range[0]))/100
    if actual_bg_fraction != bg_fraction:
        raise ValueError(f"Run {id} | Run mapped to the wrong bg_fraction. Expected {actual_bg_fraction}  but got {bg_fraction} | Run name '{name}'")
    if not config["val_dataset"] == "imagenet":
        raise ValueError(f"Run {id} | Invalid validation dataset | Expected 'imagenet' but got {config["val_dataset"]} | Run name '{name}'")
    if not is_fg_range_and_epochs_mapping_correct(config["fg_range"], config["epochs"], baseline_epochs):
        raise ValueError(f"Run {id} | FG range and epoch mapping is incorrect | FG range {config["fg_range"]} | Epoch {config["epochs"]} | Run name '{name}'")
    keywords = [config["model"].replace("/", "-"), "fornet", ("fg-" + config["fg_range"]), ("bg-" + config["bg_range"]), ("ep-" + str(config["epochs"]))]
    if config["dataset"] == "fornet/all/0.0":
        keywords.append("oip-0")
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")


def drop_columns(dataset_path:Path, column_names:list[str]):
    dataset_df = pd.read_csv(dataset_path, keep_default_na=False)
    dataset_df = dataset_df.drop(labels=column_names, axis=1, errors="ignore")
    dataset_df.to_csv(dataset_path, index=False)
    return True


def build_extreme_bg_runs_database():
    load_dotenv()
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    entity = os.getenv("WANDB_ENTITY")
    project_name = os.getenv("WANDB_PROJECT_NAME")
    org_img_probs = EXTREME_BG_ID_MAPPING.keys()
    create_database(EXTREME_BG_RUNS_FILE_PATH)
    for prob in org_img_probs:
        models = EXTREME_BG_ID_MAPPING[prob].keys()
        for model in models:
            if model not in EXTREME_BG_INSCOPE_MODELS:
                raise ValueError(
                    f"Model '{model}' is not in-scope for the Background Subset Size Ablation "
                    f"(thesis Table 4.6). Expected one of {EXTREME_BG_INSCOPE_MODELS}. "
                    f"ViT-Ti/16 is explicitly EXCLUDED from this ablation - confirmed both by "
                    f"02_EXPERIMENTAL_DESIGN.txt (Decision Log, 2026-08-16 item 5) and "
                    f"independently by thesis Table 4.6, which only lists S/16, B/16, L/16."
                )
            fg_fractions = EXTREME_BG_ID_MAPPING[prob][model].keys()
            for fg_fraction in fg_fractions:
                for bg_fraction, run_id in EXTREME_BG_ID_MAPPING[prob][model][fg_fraction].items():
                    assert isinstance(run_id, str)
                    run = api.run(f"{entity}/{project_name}/{run_id}")
                    check_ds_fraction_and_run_id_mapping(run, fg_fraction)
                    validate_extreme_bg_fornet_run(run, model, bg_fraction, prob)
                    save_run_data_to_db(run, EXTREME_BG_RUNS_FILE_PATH, extra_fields={"baseline_epochs": EXTREME_BG_BASELINE_EPOCHS})
            logger.success(f"All '{model}' model extreme BG runs (oip={prob}) are processed.")
    drop_columns(EXTREME_BG_RUNS_FILE_PATH,column_names=["crossover_epoch_val_loss", "crossover_epoch_val_acc1", "crossover_epoch_val_acc5"])


if __name__=="__main__":
    build_extreme_bg_runs_database()