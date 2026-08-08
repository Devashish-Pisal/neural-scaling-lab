import numpy as np
from types import SimpleNamespace

import pandas as pd
from src.plot_pareto import plot_pareto_frontier
from configs.config import COLUMN_NAMES
from configs.path_config import RAW_DATA_DIR
from src.plotting import *
from src.data_processing import get_imagenet_data, get_fornet_data, load_raw_file, filter_database


if __name__ == '__main__':
    csv_config = SimpleNamespace(**COLUMN_NAMES)

    # imgnet_run = filter_database(train_dataset_name="fornet/all/1.0", model_name="ViT-S/16")
    # fornet_run = filter_database(train_dataset_name="fornet/all/cos", model_name="ViT-S/16", bg_range="0-100")

    # plot_scaling_law(imgnet_run)
    # plot_scaling_law(fornet_run)

     
    # plot_dataset_size_scaling_comparison(filter_database())
    plot_fg_bg_heatmaps(filter_database())
    # plot_fornet_vs_imagenet_delta_gain(filter_database())

    # plot_crossover_epoch_vs_dataset_fraction(filter_database(), "val_loss")
    # plot_crossover_epoch_vs_dataset_fraction(filter_database(), "val_acc1")
    # plot_crossover_epoch_vs_dataset_fraction(filter_database(), "val_acc5")


    # plot_flops_scaling_comparison(filter_database())
    # plot_model_scaling_comparison(filter_database())
    # plot_pareto_frontier(filter_database())
    # plot_crossover_flops_scaling(filter_database())
