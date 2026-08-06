import pandas as pd
from pathlib import Path
from configs.path_config import RAW_DATA_DIR, MAIN_EXPERIMENT_RUNS_FILE_PATH


def load_raw_file(file_name:Path):
    results = []
    if file_name.exists() and file_name.is_file() and str(file_name).endswith(".csv"):
        df = pd.read_csv(file_name)
        results.append(df)
    return results


def process_raw_data_and_save(raw_data:list):
    pass


def get_imagenet_data(processed_data, csv_config):
    processed_data = processed_data[processed_data[csv_config.ds_name] == "ImageNet"]
    return processed_data


def get_fornet_data(processed_data, csv_config):
    processed_data = processed_data[processed_data[csv_config.ds_name] == "ForNet"]
    return processed_data


def filter_database(
        input_dataframe=None, run_id=None, run_name=None, model_name=None, train_dataset_name=None,
        train_dataset_fraction=None, fg_range=None, bg_range=None, total_epochs=None):
    database = pd.read_csv(MAIN_EXPERIMENT_RUNS_FILE_PATH)
    if input_dataframe is not None:
        database = input_dataframe
    if run_id is not None:
        database = database[database["run_id"] == run_id]
    if run_name is not None:
        database = database[database["run_name"] == run_name]
    if model_name is not None:
        database = database[database["model_name"] == model_name]
    if train_dataset_name is not None:
        database = database[database["train_dataset_name"] == train_dataset_name]
    if train_dataset_fraction is not None:
        database = database[database["train_dataset_fraction"] == train_dataset_fraction]
    if fg_range is not None:
        database = database[database["fg_range"] == fg_range]
    if bg_range is not None:
        database = database[database["bg_range"] == bg_range]
    if total_epochs is not None:
        database = database[database["total_epochs"] == total_epochs]
    return database
