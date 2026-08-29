import pandas as pd
from configs.path_config import MAIN_EXPERIMENT_RUNS_FILE_PATH, EXTREME_BG_RUNS_FILE_PATH
from src.plot_pareto import plot_pareto_frontier
from src.plotting import (
    plot_dataset_size_scaling_comparison,
    plot_fg_bg_heatmaps,
    plot_fornet_vs_imagenet_delta_gain,
    plot_model_scaling_comparison,
    plot_flops_scaling_comparison,
    plot_extreme_bg_threshold,
    plot_delta_gain_heatmap,
    plot_combined_scaling_scatter,
    plot_crossover_epoch_vs_dataset_fraction,
    plot_crossover_vs_model_size,
    plot_scaling_law,
)


if __name__ == '__main__':
    main_database = pd.read_csv(MAIN_EXPERIMENT_RUNS_FILE_PATH)
    extreme_bg_database = pd.read_csv(EXTREME_BG_RUNS_FILE_PATH)

    print("Generating thesis figures...")

    # 1. Model Scaling Comparison (Patch 16, 300-epoch baseline)
    plot_model_scaling_comparison(main_database, baseline_epochs=300)

    # 2. Dataset Size Scaling Comparison (300-epoch baseline)
    plot_dataset_size_scaling_comparison(main_database, baseline_epochs=300)

    # 3. Compute (FLOPs) Scaling Comparison (300-epoch baseline)
    plot_flops_scaling_comparison(main_database, baseline_epochs=300)

    # 4. Foreground vs Background Top-1 Accuracy Heatmaps (300-epoch baseline)
    plot_fg_bg_heatmaps(main_database, baseline_epochs=300)

    # 5. ForNet vs ImageNet Delta Gain (300-epoch baseline)
    plot_fornet_vs_imagenet_delta_gain(main_database, baseline_epochs=300)

    # 6. Delta Gain Heatmap (300-epoch baseline)
    plot_delta_gain_heatmap(main_database, metric="max_val_acc1", bg_range="0-100", baseline_epochs=300)

    # 7. Combined Scaling Overview (300-epoch baseline)
    plot_combined_scaling_scatter(main_database, bg_range="0-100", baseline_epochs=300)

    # 8. Crossover Speed Plots (300-epoch baseline)
    plot_crossover_epoch_vs_dataset_fraction(main_database, crossover_metric="val_loss", baseline_epochs=300)
    plot_crossover_vs_model_size(main_database, crossover_metric="val_loss", normalize=True, baseline_epochs=300)

    # 9. Individual Dataset Scaling Laws (300-epoch baseline)
    plot_scaling_law(main_database[main_database["train_dataset_name"] == "fornet/all/1.0"], baseline_epochs=300)
    plot_scaling_law(main_database[main_database["train_dataset_name"] == "fornet/all/cos"], baseline_epochs=300)

    # 10. Extreme Background Pool Ablation
    plot_extreme_bg_threshold(extreme_bg_database, metric="max_val_acc1")
    plot_extreme_bg_threshold(extreme_bg_database, metric="min_val_loss")

    # 11. Compute-vs-Error Pareto Frontier
    plot_pareto_frontier(main_database)

    print("All thesis figures generated and saved to outputs/!")