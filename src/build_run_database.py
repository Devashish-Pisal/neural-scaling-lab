import os
import wandb
import pandas as pd
from wandb import Run
from loguru import logger
from dotenv import load_dotenv
from configs.path_config import MAIN_DATABASE_PATH
from configs.config import WANDB_RUN_CONFIG, DB_COLUMNS, EXPERIMENT_CONSTANTS, FG_RANGE_COUNT_MAPPING, BG_RANGE_COUNT_MAPPING



def validate_imgnet_run(run:Run, expected_model:str):
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
    if config["epochs"] not in (300, 396, 522, 754):
        raise ValueError(f"Run {id} | Invalid Epochs | Expected (300, 396, 522, 754) but got {config["epochs"]} | Run name '{name}'")
    if config["fg_range"] not in ("0-10", "0-25", "0-50", "0-100"):
        raise ValueError(f"Run {id} | Invalid FG range provided | Expected ('0-10', '0-25', '0-50', '0-100') but got {config["fg_range"]} | Run name '{name}'")
    if config["bg_range"] is not None:
        raise ValueError(f"Run {id} | Invalid BG range | Expected 'None' but got {config["bg_range"]} | Run name '{name}'")
    if not config["val_dataset"] == "imagenet":
        raise ValueError(f"Run {id} | Invalid validation dataset | Expected 'imagenet' but got {config["val_dataset"]} | Run name '{name}'")
    if "bg" in str(name):
        raise ValueError(f"Run {id} | Invalid run name. Run name for imagenet dataset should not contain 'bg' | Run name '{name}'")
    if not is_fg_range_and_epochs_mapping_correct(config["fg_range"], config["epochs"]):
        raise ValueError(f"Run {id} | FG range and epoch mapping is incorrect | FG range {config["fg_range"]} | Epoch {config["epochs"]} | Run name '{name}'")
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")


def validate_fornet_run(run:Run, expected_model:str):
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
    if config["epochs"] not in (300, 396, 522, 754):
        raise ValueError(f"Run {id} | Invalid Epochs | Expected (300, 396, 522, 754) but got {config["epochs"]} | Run name '{name}'")
    if config["fg_range"] not in ("0-10", "0-25", "0-50", "0-100"):
        raise ValueError(f"Run {id} | Invalid FG range provided | Expected ('0-10', '0-25', '0-50', '0-100') but got {config["fg_range"]} | Run name '{name}'")
    if config["bg_range"] not in ("0-10", "0-25", "0-50", "0-100"):
        raise ValueError(f"Run {id} | Invalid BG range | Expected ('0-10', '0-25', '0-50', '0-100') but got {config["bg_range"]} | Run name '{name}'")
    if not config["val_dataset"] == "imagenet":
        raise ValueError(f"Run {id} | Invalid validation dataset | Expected 'imagenet' but got {config["val_dataset"]} | Run name '{name}'")
    if not is_fg_range_and_epochs_mapping_correct(config["fg_range"], config["epochs"]):
        raise ValueError(f"Run {id} | FG range and epoch mapping is incorrect | FG range {config["fg_range"]} | Epoch {config["epochs"]} | Run name '{name}'")
    keywords = [config["model"].replace("/", "-"), "fornet", ("fg-" + config["fg_range"]), ("bg-" + config["bg_range"]), ("ep-" + str(config["epochs"]))]
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")




def is_fg_range_and_epochs_mapping_correct(fg_range, epoch):
    return (
        (fg_range == "0-10" and epoch == 754) or
        (fg_range == "0-25" and epoch == 522) or
        (fg_range == "0-50" and epoch == 396) or
        (fg_range == "0-100" and epoch == 300)
    )


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


def create_main_database():
    if not MAIN_DATABASE_PATH.exists():
        MAIN_DATABASE_PATH.touch()
        logger.info(f"Database file created at path '{MAIN_DATABASE_PATH}'")
        with open(MAIN_DATABASE_PATH, "w", encoding="utf-8") as file:
            file.write(",".join(DB_COLUMNS.keys()))
    else:
        logger.info(f"Database file already exists at path '{MAIN_DATABASE_PATH}'")


def perform_column_operation(column:pd.Series, operation:str):
    match operation:
        case "max": return column.max()
        case "min": return column.min()
        case "final": return column.iloc[-1]
    raise ValueError(f"Operation '{operation}' is not implemented")


def save_run_data_to_db(run:Run, dataset:str):
    database_df = pd.read_csv(MAIN_DATABASE_PATH, keep_default_na=False)
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
        row["train_dataset_fraction"] = (end-start)/100

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
        row["flops_per_epoch"] = calculate_flops_per_epoch(config["model"].strip().lower(), row["steps_per_epoch"])
        # Formula 1 for total_flops: C = 6 * N * (total_steps * batch_size)
        # Formula 2 for total_flops: C = 6 * N * (train_dataset_size * epochs)
        row["total_flops"] = row["flops_per_epoch"] * row["total_epochs"]
        # modify crossover values later, once all run for one model are finished
        row["crossover_epoch_val_loss"] = -1
        row["crossover_epoch_val_acc1"] = -1
        row["crossover_epoch_val_acc5"] = -1
        row["total_runtime"] = perform_column_operation(history["_runtime"], "final")
        row["gpu_partition"] = metadata["gpu"]

        new_row = pd.DataFrame([row], columns=DB_COLUMNS.keys())
        df_combined = pd.concat([database_df, new_row], axis=0)
        df_combined.to_csv(MAIN_DATABASE_PATH, index=False)
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
        first_true_idx = comparison.idxmax()
        return int(merged.loc[first_true_idx, "epoch"])


def find_crossover_epochs(imgnet_runs_dict:dict, fornet_runs_dict:dict, api, entity, project_name):
    assert len(imgnet_runs_dict) == 4 == len(fornet_runs_dict)
    database_df = pd.read_csv(MAIN_DATABASE_PATH, keep_default_na=False)
    for fraction, imgnet_run_id in imgnet_runs_dict.items():
        # only consider fornet run which has bg_range = '0-100'
        fornet_run_id = fornet_runs_dict[fraction][0]
        imgnet_run = api.run(f"{entity}/{project_name}/{imgnet_run_id[0]}")
        fornet_run = api.run(f"{entity}/{project_name}/{fornet_run_id}")
        imgnet_config = imgnet_run.config
        fornet_config = fornet_run.config
        assert fornet_config["dataset"] == "fornet/all/cos" and fornet_config["bg_range"] == "0-100"
        assert imgnet_config["dataset"] == "fornet/all/1.0" and imgnet_config["bg_range"] is None
        assert imgnet_config["fg_range"] == fornet_config["fg_range"]

        imgnet_run_history = pd.DataFrame(imgnet_run.scan_history())
        fornet_run_history = pd.DataFrame(fornet_run.scan_history())
        database_df.loc[database_df["run_id"] == imgnet_run_id[0], "crossover_epoch_val_loss"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/loss")
        database_df.loc[database_df["run_id"] == imgnet_run_id[0], "crossover_epoch_val_acc1"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/acc1")
        database_df.loc[database_df["run_id"] == imgnet_run_id[0], "crossover_epoch_val_acc5"] = find_crossover_point(imgnet_run_history, fornet_run_history, "val/acc5")
    database_df.to_csv(MAIN_DATABASE_PATH, index=False)




def build_database():
    load_dotenv()
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    entity = os.getenv("WANDB_ENTITY")
    project_name = os.getenv("WANDB_PROJECT_NAME")
    models = WANDB_RUN_CONFIG.keys()
    create_main_database()
    for model in models:
        imgnet_runs_dict = WANDB_RUN_CONFIG[model]["imagenet_run_ids"]
        fornet_runs_dict = WANDB_RUN_CONFIG[model]["fornet_run_ids"]
        for fraction, ids in imgnet_runs_dict.items():
            assert len(ids) == 1
            run = api.run(f"{entity}/{project_name}/{ids[0]}")
            check_ds_fraction_and_run_id_mapping(run, fraction)
            validate_imgnet_run(run, model)
            save_run_data_to_db(run, run.config["dataset"])
        for fraction, ids in fornet_runs_dict.items():
            assert 0 < len(ids) < 5
            for run_id in ids:
                run = api.run(f"{entity}/{project_name}/{run_id}")
                check_ds_fraction_and_run_id_mapping(run, fraction)
                validate_fornet_run(run, model)
                save_run_data_to_db(run, run.config["dataset"])
        find_crossover_epochs(imgnet_runs_dict, fornet_runs_dict, api, entity, project_name)
        logger.success(f"All '{model}' model model runs are processed.")



if __name__=="__main__":
    build_database()


