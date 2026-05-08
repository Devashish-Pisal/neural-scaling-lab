import pandas as pd
from pathlib import Path
from configs.path_config import RAW_DATA_DIR


def load_raw_data():
    results = []
    for item in RAW_DATA_DIR.iterdir():
        if item.is_file() and str(item).endswith(".csv"):
            df = pd.read_csv(item)
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