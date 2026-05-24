import os
import wandb
import pandas as pd
from wandb import Run
from loguru import logger
from dotenv import load_dotenv
from pprint import pprint
from configs.path_config import MAIN_DATABASE_PATH
from configs.config import WANDB_RUN_CONFIG, DB_COLUMNS



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
    keywords = [config["model"].replace("/", "-"), "fornet", ("fg-" + config["fg_range"]), ("bg-" + config["bg_range"]), ("ep-" + str(config["epochs"]))]
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
    if not contains_run_name_keywords(name, keywords):
        raise ValueError(f"Run {id} | Keywords are not matching with run name | Keywords {keywords} | Run name '{name}'")




def is_fg_range_and_epochs_mapping_correct(fg_range, epoch):
    return (
        (fg_range == "0-10" and epoch == 754) or
        (fg_range == "0-25" and epoch == 522) or
        (fg_range == "0-50" and epoch == 396) or
        (fg_range == "0-100" and epoch == 300)
    )


def contains_run_name_keywords(run_name:str, keywords:list[str]):
    run_name = run_name.strip().lower()
    keywords = [keyword.strip().lower() for keyword in keywords]
    for kw in keywords:
        if not kw in run_name:
            print(f"Run Name: {run_name} | Keyword: {kw}")
            return False
    return True


def create_main_database():
    if not MAIN_DATABASE_PATH.exists():
        MAIN_DATABASE_PATH.touch()
        logger.info(f"Database file created at path '{MAIN_DATABASE_PATH}'")
        with open(MAIN_DATABASE_PATH, "w", encoding="utf-8") as file:
            file.write(",".join(DB_COLUMNS.keys()))
    else:
        logger.info(f"Database file already exists at path '{MAIN_DATABASE_PATH}'")


def

def save_run_data_to_db(run:Run, dataset:str):
    database_df = pd.read_csv(MAIN_DATABASE_PATH)
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
        row["bg_range"] = config["bg_range"]
        row["fg_count"] = config["fg_count"]
        row["bg_count"] = config["bg_count"]
        row["train_dataset_size"] = config["fg_count"]
        row["train_dataset_fraction"] = None
        row["total_epochs"] = config["epochs"]

        row["min_train_loss"] = None

    else:
        logger.info(f"Data of run {run.id} is already stored in database | Run name '{run.name}'")





if __name__=="__main__":
    load_dotenv()
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    entity = os.getenv("WANDB_ENTITY")
    project_name = os.getenv("WANDB_PROJECT_NAME")
    models = WANDB_RUN_CONFIG.keys()
    create_main_database()

    for model in models:
        imgnet_runs = WANDB_RUN_CONFIG[model]["imagenet_run_ids"]
        fornet_runs = WANDB_RUN_CONFIG[model]["fornet_run_ids"]

        for run_id in imgnet_runs:
            run = api.run(f"{entity}/{project_name}/{run_id}")
            validate_imgnet_run(run, model)
            save_run_data_to_db(run, run.config["dataset"])
        continue
        for run_id in fornet_runs:
            run = api.run(f"{entity}/{project_name}/{run_id}")
            validate_fornet_run(run, model)
            save_run_data_to_db(run, run.config["dataset"])
        logger.info(f"All '{model}' model model runs are processed.")


