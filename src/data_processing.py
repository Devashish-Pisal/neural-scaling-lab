import pandas as pd
from pathlib import Path
from configs.path_config import RAW_DATA_DIR


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