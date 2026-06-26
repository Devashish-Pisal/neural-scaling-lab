import numpy as np
from types import SimpleNamespace
from configs.config import COLUMN_NAMES
from configs.path_config import RAW_DATA_DIR
from src.plotting import *
from src.data_processing import get_imagenet_data, get_fornet_data, load_raw_file, filter_database


if __name__ == '__main__':
    csv_config = SimpleNamespace(**COLUMN_NAMES)

    '''
    output_file_path = RAW_DATA_DIR / "steps_scaling_exponent-0.6_output.csv"
    scaling_exponent = 0.6

    result = load_raw_file(output_file_path)
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
    plot_steps_allocation(ds_samples, imgnet_steps, "ImageNet", scaling_exponent)
    plot_steps_allocation(ds_samples, fornet_steps, "ForNet", scaling_exponent)
    plot_scaling_law(ds_samples, imgnet_val_loss, "ImageNet")
    plot_scaling_law(ds_samples, fornet_val_loss, "ForNet")
    plot_dataset_size_scaling_comparison(ds_samples, imgnet_val_loss, fornet_val_loss)
    plot_flops_scaling_comparison(flops, imgnet_val_loss, fornet_val_loss)

    plot_acc_comparison(ds_samples, imgnet_top_1_acc, fornet_top_1_acc, "Top-1")
    plot_acc_comparison(ds_samples, imgnet_top_5_acc, fornet_top_5_acc, "Top-5")

    plot_compute_efficiency_comparison(flops, imgnet_top_1_acc, fornet_top_1_acc, "Top-1")
    plot_compute_efficiency_comparison(flops, imgnet_top_5_acc, fornet_top_5_acc, "Top-5")

    plot_sample_efficiency_comparison(ds_samples, imgnet_top_1_acc, fornet_top_1_acc, "Top-1", 0.70)
    plot_sample_efficiency_comparison(ds_samples, imgnet_top_5_acc, fornet_top_5_acc, "Top-5", 0.90)
    '''

    '''
    scaling_exponents = [0.6, 0.8, 1.0]
    all_results = {}
    for exponent in scaling_exponents:
        file_path = (
            RAW_DATA_DIR /
            f"steps_scaling_exponent-{exponent}_output.csv"
        )
        df = load_raw_file(file_path)[0]
        all_results[exponent] = {
            "imagenet": get_imagenet_data(df, csv_config),
            "fornet": get_fornet_data(df, csv_config)
        }
    set_style()
 
    plot_exponent_scaling_consistency(
        all_results,
        csv_config,
        "ImageNet"
    )
    plot_exponent_scaling_consistency(
        all_results,
        csv_config,
        "ForNet"
    )


    plot_exponent_fit_quality(
        all_results,
        csv_config
    )


    plot_exponent_compute_efficiency(
        all_results,
        csv_config,
        "ImageNet"
    )
    plot_exponent_compute_efficiency(
        all_results,
        csv_config,
        "ForNet"
    )
   '''
    imgnet_run = filter_database(train_dataset_name="fornet/all/1.0")
    fornet_run = filter_database(train_dataset_name="fornet/all/cos")
    plot_scaling_law(imgnet_run)
    plot_scaling_law(fornet_run)

     
    # plot_dataset_size_scaling_comparison(filter_database())

    # plot_fg_bg_heatmaps(filter_database())

    # plot_fornet_vs_imagenet_delta_gain(filter_database())
    # plot_crossover_flops_scaling(filter_database())
