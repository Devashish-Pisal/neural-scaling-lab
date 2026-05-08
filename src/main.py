import numpy as np
from types import SimpleNamespace
from configs.csv_file_config import COLUMN_NAMES
from src.plotting import *
from src.data_processing import get_imagenet_data, get_fornet_data, load_raw_data


if __name__ == '__main__':
    csv_config = SimpleNamespace(**COLUMN_NAMES)

    result = load_raw_data()
    imgnet_data = get_imagenet_data(result[0],csv_config)
    fornet_data = get_fornet_data(result[0], csv_config)
    ds_samples = imgnet_data[csv_config.real_ds_samples]  # fornet_data.real_ds_samples also works (both are same)
    flops = imgnet_data[csv_config.real_FLOPs] # fornet_data.real_flops also works (both are same)

    imgnet_steps = imgnet_data[csv_config.real_steps]
    fornet_steps = fornet_data[csv_config.real_steps]

    imgnet_val_loss =  imgnet_data[csv_config.min_val_loss]
    fornet_val_loss = fornet_data[csv_config.min_val_loss]

    imgnet_top_1_acc = imgnet_data[csv_config.max_val_acc1]
    fornet_top_1_acc = fornet_data[csv_config.max_val_acc1]

    imgnet_top_5_acc = imgnet_data[csv_config.max_val_acc5]
    fornet_top_5_acc = fornet_data[csv_config.max_val_acc5]

    set_style()
    # plot_scaling_law(ds_samples, imgnet_val_loss, "ImageNet")
    # plot_scaling_law(ds_samples, fornet_val_loss, "ForNet")
    # plot_dataset_size_scaling_comparison(ds_samples, imgnet_val_loss, fornet_val_loss)
    # plot_acc_comparison(ds_samples, imgnet_top_1_acc, fornet_top_1_acc, "Top-1")
    # plot_acc_comparison(ds_samples, imgnet_top_5_acc, fornet_top_5_acc, "Top-5")
    # plot_steps_allocation(ds_samples, imgnet_steps, "ImageNet")
    # plot_steps_allocation(ds_samples, fornet_steps, "ForNet")
    # plot_flops_scaling_comparison(flops, imgnet_val_loss, fornet_val_loss)
    # plot_compute_efficiency_comparison(flops, imgnet_top_1_acc, fornet_top_1_acc, "Top-1")
    # plot_compute_efficiency_comparison(flops, imgnet_top_5_acc, fornet_top_5_acc, "Top-5")
    # plot_sample_efficiency_comparison(ds_samples, imgnet_top_1_acc, fornet_top_1_acc, "Top-1", 0.70)
    # plot_sample_efficiency_comparison(ds_samples, imgnet_top_5_acc, fornet_top_5_acc, "Top-5", 0.90)



