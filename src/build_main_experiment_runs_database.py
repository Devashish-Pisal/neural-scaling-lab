import os
import wandb
import pandas as pd
from wandb import Run
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
from configs.path_config import MAIN_EXPERIMENT_RUNS_FILE_PATH
from configs.metric import METRICS
from configs.config import (
    MAIN_EXPERIMENT_ID_MAPPING,
    DB_COLUMNS,
    EXPERIMENT_CONSTANTS,
    FG_RANGE_COUNT_MAPPING,
    BG_RANGE_COUNT_MAPPING,
    BASELINE_EPOCH_SCHEDULES,
    FG_BG_GRID_MODELS,
)



def validate_imgnet_run(run:Run, expected_model:str, baseline_epochs:int = 300):
    id = run.id
    name = run.name
    config = run.config
    keywords = [config["model"].replace("/", "-"), "imgnet", ("fg-" + config["fg_range"]), ("ep-" + str(config["epochs"]))]
    if run.state != "finished":
        raise ValueError(f"Run {id} | Run state is not 'finished'. Expected run state 'finished' but got '{run.state}' | Run name '{name}'")
    if config["model"].lower().replace("/", "-") != expected_model.lower().replace("/", "-"):
        raise ValueError(f"Run {id} | Model mismatch. Expected '{expected_model}' but got '{config['model']}'")
    if not "imgnet" in str(name):
        raise ValueError(f"Run {id} | Run does not contains 'imgnet' in its run name | Run name '{name}'")
    if not config["dataset"] == "fornet/all/1.0":
        raise ValueError(f"Run {id} | Dataset mismatch. Expected 'fornet/all/1.0' but got {config["dataset"]} | Run name '{name}'")
    valid_epochs = tuple(BASELINE_EPOCH_SCHEDULES[baseline_epochs].values())
    if config["epochs"] not in valid_epochs:
        raise ValueError(f"Run {id} | Invalid Epochs for baseline_epochs={baseline_epochs} | Expected {valid_epochs} but got {config["epochs"]} | Run name '{name}'")
    if config["fg_range"] not in ("0-10", "0-25", "0-50", "0-100"):
        raise ValueError(f"Run {id} | Invalid FG range provided | Expected ('0-10', '0-25', '0-50', '0-100') but got {config["fg_range"]} | Run name '{name}'")
    if config["bg_range"] is not None:
        raise ValueError(f"Run {id} | Invalid BG range | Expected 'None' but got {config["bg_range"]} | Run name '{name}'")
    if not config["val_dataset"] == "imagenet":
        raise ValueError(f"Run {id} | Invalid validation dataset | Expected 'imagenet' but got {config["val_dataset"]} | Run name '{name}'")
    if "bg" in str(name):
        raise ValueError(f"Run {id} | Invalid run name. Run name for imagenet dataset should not contain 'bg' | Run name '{name}'")
    if not is_fg_range_and_epochs_mapping_correct(config["fg_range"], config["epochs"], baseline_epochs):
        raise ValueError(f"Run {id} | FG range and epoch mapping is incorrect for baseline_epochs={baseline_epochs} | FG range {config["fg_range"]} | Epoch {config["epochs"]} | Run name '{name}'")
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")


def validate_fornet_run(run:Run, expected_model:str, baseline_epochs:int = 300, allowed_bg_ranges:tuple = ("0-10", "0-25", "0-50", "0-100")):
    id = run.id
    name = run.name
    config = run.config
    if run.state != "finished":
        raise ValueError(f"Run {id} | Run state is not 'finished'. Expected run state 'finished' but got '{run.state}' | Run name '{name}'")
    if config["model"].lower().replace("/", "-") != expected_model.lower().replace("/", "-"):
        raise ValueError(f"Run {id} | Model mismatch. Expected '{expected_model}' but got '{config['model']}'")
    if not "fornet" in str(name):
        raise ValueError(f"Run {id} | Run does not contains 'fornet' in its run name | Run name '{name}'")
    if not config["dataset"] == "fornet/all/cos":
        raise ValueError(f"Run {id} | Dataset mismatch. Expected 'fornet/all/cos' but got {config["dataset"]} | Run name '{name}'")
    valid_epochs = tuple(BASELINE_EPOCH_SCHEDULES[baseline_epochs].values())
    if config["epochs"] not in valid_epochs:
        raise ValueError(f"Run {id} | Invalid Epochs for baseline_epochs={baseline_epochs} | Expected {valid_epochs} but got {config["epochs"]} | Run name '{name}'")
    if config["fg_range"] not in ("0-10", "0-25", "0-50", "0-100"):
        raise ValueError(f"Run {id} | Invalid FG range provided | Expected ('0-10', '0-25', '0-50', '0-100') but got {config["fg_range"]} | Run name '{name}'")
    if config["bg_range"] not in allowed_bg_ranges:
        raise ValueError(f"Run {id} | Invalid BG range | Expected {allowed_bg_ranges} but got {config["bg_range"]} | Run name '{name}'")
    if not config["val_dataset"] == "imagenet":
        raise ValueError(f"Run {id} | Invalid validation dataset | Expected 'imagenet' but got {config["val_dataset"]} | Run name '{name}'")
    if not is_fg_range_and_epochs_mapping_correct(config["fg_range"], config["epochs"], baseline_epochs):
        raise ValueError(f"Run {id} | FG range and epoch mapping is incorrect for baseline_epochs={baseline_epochs} | FG range {config["fg_range"]} | Epoch {config["epochs"]} | Run name '{name}'")
    keywords = [config["model"].replace("/", "-"), "fornet", ("fg-" + config["fg_range"]), ("bg-" + config["bg_range"]), ("ep-" + str(config["epochs"]))]
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")




def is_fg_range_and_epochs_mapping_correct(fg_range, epoch, baseline_epochs=300):
    schedule = BASELINE_EPOCH_SCHEDULES.get(baseline_epochs)
    if schedule is None:
        raise ValueError(
            f"Unknown baseline_epochs '{baseline_epochs}'. Expected one of "
            f"{sorted(BASELINE_EPOCH_SCHEDULES.keys())} (see configs/config.py, BASELINE_EPOCH_SCHEDULES)."
        )
    return schedule.get(fg_range) == epoch


def check_ds_fraction_and_run_id_mapping(run:Run, ds_fraction_from_mapping:float):
    actual_fg_range = run.config["fg_range"]
    output = ((actual_fg_range == "0-10" and ds_fraction_from_mapping == 0.10) or
              (actual_fg_range == "0-25" and ds_fraction_from_mapping == 0.25) or
              (actual_fg_range == "0-50" and ds_fraction_from_mapping == 0.50) or
              (actual_fg_range == "0-100" and ds_fraction_from_mapping == 1.00)
              )
    if not output:
        raise ValueError(f"Fraction - ID mapping for run id {run.id} is incorrect in config.py. For {actual_fg_range} mapped {ds_fraction_from_mapping}"
                         f" | Run name {run.name}")


def contains_run_name_keywords(run_name:str, keywords:list[str]):
    run_name = run_name.strip().lower()
    keywords = [keyword.strip().lower() for keyword in keywords]
    for kw in keywords:
        if not kw in run_name:
            print(f"Run Name: {run_name} | Keyword: {kw}")
            return False
    return True


def calculate_flops_per_epoch(model_name, steps_per_epoch):
    model_param_count = EXPERIMENT_CONSTANTS["model_parameters"][model_name]
    model_name = model_name.split("/")
    assert len(model_name) == 2
    # patch_size = int(model_name[1])
    return 6 * model_param_count * EXPERIMENT_CONSTANTS["global_batch_size"] * steps_per_epoch


def create_database(database_file_path: Path, columns: dict = None):
    columns = columns if columns is not None else DB_COLUMNS
    if not database_file_path.exists():
        database_file_path.touch()
        logger.info(f"Database file created at path '{database_file_path}'")
        with open(database_file_path, "w", encoding="utf-8") as file:
            file.write(",".join(columns.keys()))
    else:
        logger.info(f"Database file already exists at path '{database_file_path}'")


def perform_column_operation(column:pd.Series, operation:str):
    match operation:
        case "max": return column.max()
        case "min": return column.min()
        case "final":
            valid = column.dropna()
            return valid.iloc[-1] if not valid.empty else float("nan")
    raise ValueError(f"Operation '{operation}' is not implemented")


def save_run_data_to_db(run:Run, dataset_path:Path, extra_fields: dict = None):
    database_df = pd.read_csv(dataset_path, keep_default_na=False)
    if not run.id in database_df["run_id"].values:
        config = run.config  # dict
        metadata = run.metadata  # dict
        history = pd.DataFrame(run.scan_history()) # dataframe
        row = DB_COLUMNS.copy()
        row["run_id"] = run.id
        row["run_name"] = run.name
        row["model_name"] = config["model"]
        row["train_dataset_name"] = config["dataset"]
        row["fg_range"] = config["fg_range"]
        row["bg_range"] = "null" if config["bg_range"] is None else config["bg_range"]
        row["fg_count"] = FG_RANGE_COUNT_MAPPING[config["fg_range"]]
        row["bg_count"] = BG_RANGE_COUNT_MAPPING[row["bg_range"]]
        row["train_dataset_size"] = row["fg_count"]

        fg_range_list = config["fg_range"].split("-")
        start = int(fg_range_list[0])
        end = int(fg_range_list[1])
        row["train_dataset_fraction"] = (end-start)/100  # alternative colum name fg_fraction

        calc_bg_fraction = lambda start_and_end: (float(start_and_end[1]) - float(start_and_end[0]))/100
        row["bg_fraction"] = "null" if config["bg_range"] is None else calc_bg_fraction(config["bg_range"].split("-"))

        row["total_epochs"] = config["epochs"]
        row["min_train_loss"] = perform_column_operation(history["train/loss"], "min")
        row["min_train_loss_epoch"] = history.loc[history["train/loss"].idxmin(), "epoch"]
        row["min_val_loss"] = perform_column_operation(history["val/loss"], "min")
        row["min_val_loss_epoch"] =  history.loc[history["val/loss"].idxmin(), "epoch"]
        row["max_val_acc1"] = perform_column_operation(history["val/acc1"], "max")
        row["max_val_acc1_epoch"] = history.loc[history["val/acc1"].idxmax(), "epoch"]
        row["max_val_acc5"] = perform_column_operation(history["val/acc5"], "max")
        row["max_val_acc5_epoch"] =  history.loc[history["val/acc5"].idxmax(), "epoch"]
        row["final_val_loss"] = perform_column_operation(history["val/loss"], "final")
        row["final_val_acc1"] = perform_column_operation(history["val/acc1"], "final")
        row["final_val_acc5"] = perform_column_operation(history["val/acc5"], "final")
        row["final_train_loss"] = perform_column_operation(history["train/loss"], "final")
        row["steps_per_epoch"] = int(row["fg_count"] / EXPERIMENT_CONSTANTS["global_batch_size"])
        row["total_steps"] = int(row["steps_per_epoch"] * config["epochs"])

        row["parameter_count"] = METRICS[config["model"].strip().lower()]["eval/number of parameters"]
        row["macs_per_image"] = METRICS[config["model"].strip().lower()]["eval/macs"]
        row["flops_per_image"] = METRICS[config["model"].strip().lower()]["eval/flops"]
        row["total_flops"] = 3 * row["flops_per_image"] * row["train_dataset_size"] * row["total_epochs"]

        # modify crossover values later, once all run for one model are finished
        row["crossover_epoch_val_loss"] = -1
        row["crossover_epoch_val_acc1"] = -1
        row["crossover_epoch_val_acc5"] = -1
        row["total_runtime"] = perform_column_operation(history["_runtime"], "final")
        row["gpu_partition"] = metadata.get("gpu", None) if metadata else None

        if extra_fields:
            row.update(extra_fields)

        new_row = pd.DataFrame([row], columns=list(row.keys()))
        df_combined = pd.concat([database_df, new_row], axis=0)
        df_combined.to_csv(dataset_path, index=False)
        logger.info(f"New row written to database | Row {row}")
    else:
        logger.info(f"Data of run {run.id} is already stored in database | Run name '{run.name}'")


def find_crossover_point(imgnet_df: pd.DataFrame, fornet_df: pd.DataFrame, column_name):
        # Filter out NaNs and duplicates for epoch and the target column
        imgnet_val = imgnet_df[["epoch", column_name]].dropna().drop_duplicates(subset=["epoch"])
        fornet_val = fornet_df[["epoch", column_name]].dropna().drop_duplicates(subset=["epoch"])
        # Merge on epoch to align them
        merged = pd.merge(imgnet_val, fornet_val, on="epoch", suffixes=("_imgnet", "_fornet"))
        merged = merged.sort_values("epoch").reset_index(drop=True)
        if merged.empty:
            return -1
        if column_name == "val/loss":
            comparison = merged[f"{column_name}_imgnet"] > merged[f"{column_name}_fornet"]
        elif column_name in ("val/acc1", "val/acc5"):
            comparison = merged[f"{column_name}_imgnet"] < merged[f"{column_name}_fornet"]
        else:
            raise ValueError(f"Wrong column name {column_name} provided")
        if not comparison.any():
            return -1
        comparison.iloc[0:5] = False  # Ignore first 5 epochs, as they are warmup epochs don't consider them for crossover
        first_true_idx = comparison.idxmax()
        return int(merged.loc[first_true_idx, "epoch"])


def find_crossover_epochs(imgnet_runs_dict:dict, fornet_runs_dict:dict, api, entity, project_name, database_file_path: Path = MAIN_EXPERIMENT_RUNS_FILE_PATH):
    assert len(imgnet_runs_dict) == 4 == len(fornet_runs_dict)
    database_df = pd.read_csv(database_file_path, keep_default_na=False)
    for fraction, imgnet_run_id in imgnet_runs_dict.items():
        imgnet_run = api.run(f"{entity}/{project_name}/{imgnet_run_id[0]}")
        imgnet_config = imgnet_run.config
        assert imgnet_config["dataset"] == "fornet/all/1.0" and imgnet_config["bg_range"] is None
        fornet_bg_fractions = fornet_runs_dict[fraction]
        for fornet_run_id in fornet_bg_fractions:
            fornet_run = api.run(f"{entity}/{project_name}/{fornet_run_id}")
            fornet_config = fornet_run.config
            assert imgnet_config["fg_range"] == fornet_config["fg_range"]
            assert fornet_config["dataset"] == "fornet/all/cos"
            imgnet_run_history = pd.DataFrame(imgnet_run.scan_history())
            fornet_run_history = pd.DataFrame(fornet_run.scan_history())
            database_df.loc[database_df["run_id"] == fornet_run_id, "crossover_epoch_val_loss"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/loss")
            database_df.loc[database_df["run_id"] == fornet_run_id, "crossover_epoch_val_acc1"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/acc1")
            database_df.loc[database_df["run_id"] == fornet_run_id, "crossover_epoch_val_acc5"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/acc5")
    database_df.to_csv(database_file_path, index=False)


def _resolve_fornet_allowed_bg_ranges(model:str, baseline_epochs:int, fraction:float, ids:list):
    """Determine which bg_range values are valid for a given
    (model, baseline_epochs, fraction) fornet_run_ids entry, and assert the
    entry's shape is consistent with thesis Chapter 4:
      - len(ids) == 4  -> full fg x bg grid (Table 4.5). Only legal for
        baseline_epochs == 300 and model in FG_BG_GRID_MODELS.
      - len(ids) == 1  -> Scaling Law Analysis only (Table 4.4), bg fixed
        at "0-100". Legal for any model at any baseline.
    Any other length is a config.py data-entry error and raises loudly
    rather than silently mis-validating.
    """
    if len(ids) == 4:
        if baseline_epochs != 300 or model not in FG_BG_GRID_MODELS:
            raise ValueError(
                f"Model '{model}' baseline_epochs={baseline_epochs} fraction={fraction}: "
                f"a 4-element fornet_run_ids list (full fg x bg grid) is only valid for "
                f"baseline_epochs=300 and models in FG_BG_GRID_MODELS={FG_BG_GRID_MODELS} "
                f"(thesis Table 4.5 / 02_EXPERIMENTAL_DESIGN.txt Group 2a). Check "
                f"configs/config.py, MAIN_EXPERIMENT_ID_MAPPING for a possible typo."
            )
        return ("0-10", "0-25", "0-50", "0-100")
    if len(ids) == 1:
        return ("0-100",)
    raise ValueError(
        f"Model '{model}' baseline_epochs={baseline_epochs} fraction={fraction}: "
        f"fornet_run_ids must have exactly 1 (Scaling Law Analysis, bg=0-100 fixed) or "
        f"4 (full fg x bg grid) entries, got {len(ids)}. Check configs/config.py."
    )


def build_main_experiment_runs_database():
    load_dotenv()
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    entity = os.getenv("WANDB_ENTITY")
    project_name = os.getenv("WANDB_PROJECT_NAME")
    create_database(MAIN_EXPERIMENT_RUNS_FILE_PATH)

    for model, baselines in MAIN_EXPERIMENT_ID_MAPPING.items():
        for baseline_epochs, runs in baselines.items():
            imgnet_runs_dict = runs.get("imagenet_run_ids", {})
            fornet_runs_dict = runs.get("fornet_run_ids", {})

            if not imgnet_runs_dict and not fornet_runs_dict:
                logger.info(f"Skipping '{model}' baseline_epochs={baseline_epochs} - no run IDs filled in yet.")
                continue

            for fraction, ids in imgnet_runs_dict.items():
                assert len(ids) == 1, f"'{model}' baseline_epochs={baseline_epochs}: imagenet_run_ids[{fraction}] must have exactly 1 id, got {len(ids)}"
                run = api.run(f"{entity}/{project_name}/{ids[0]}")
                check_ds_fraction_and_run_id_mapping(run, fraction)
                validate_imgnet_run(run, model, baseline_epochs=baseline_epochs)
                save_run_data_to_db(run, MAIN_EXPERIMENT_RUNS_FILE_PATH, extra_fields={"baseline_epochs": baseline_epochs})

            for fraction, ids in fornet_runs_dict.items():
                allowed_bg_ranges = _resolve_fornet_allowed_bg_ranges(model, baseline_epochs, fraction, ids)
                for run_id in ids:
                    run = api.run(f"{entity}/{project_name}/{run_id}")
                    check_ds_fraction_and_run_id_mapping(run, fraction)
                    validate_fornet_run(run, model, baseline_epochs=baseline_epochs, allowed_bg_ranges=allowed_bg_ranges)
                    save_run_data_to_db(run, MAIN_EXPERIMENT_RUNS_FILE_PATH, extra_fields={"baseline_epochs": baseline_epochs})

            if len(imgnet_runs_dict) == 4 and len(fornet_runs_dict) == 4:
                find_crossover_epochs(imgnet_runs_dict, fornet_runs_dict, api, entity, project_name, database_file_path=MAIN_EXPERIMENT_RUNS_FILE_PATH)
            else:
                logger.info(
                    f"Skipping crossover computation for '{model}' baseline_epochs={baseline_epochs} - "
                    f"need all 4 fractions on both sides (have {len(imgnet_runs_dict)} imagenet, "
                    f"{len(fornet_runs_dict)} fornet)."
                )

            logger.success(f"All '{model}' model runs for baseline_epochs={baseline_epochs} are processed.")



if __name__=="__main__":
    build_main_experiment_runs_database()