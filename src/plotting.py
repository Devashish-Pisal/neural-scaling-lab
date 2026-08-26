import math
import numpy as np
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from src.data_processing import filter_database
from configs.path_config import IMG_OUTPUT_DIR, PDF_OUTPUTS_DIR
from configs.config import EXPERIMENT_CONSTANTS
from matplotlib.lines import Line2D


_MODEL_SIZE_ORDER = ["ViT-Ti/16", "ViT-S/16", "ViT-S/32", "ViT-B/16", "ViT-B/28", "ViT-L/16", "ViT-L/32"]
# _MODEL_SIZE_ORDER = ["ViT-Ti/16", "ViT-S/16", "ViT-B/16", "ViT-L/16"]
_DATASET_LABELS = {
    "fornet/all/cos": ("Cosine mixing", "o", "#1f77b4"),
    "fornet/all/0.0": ("No mixing (0.0)", "s", "#d62728"),
}



def plot_dataset_size_scaling_comparison(df: DataFrame):
    set_style()

    df = df[df["bg_range"].isin(["0-100", None])].copy()
    df["architecture"] = df["model_name"].str.split("/").str[0]
    df["patch_size"] = df["model_name"].str.split("/").str[1]

    architectures = sorted(df["architecture"].unique())

    patch_colors = {"16": "#1f77b4", "28": "#2ca02c", "32": "#d62728"}
    datasets = {
        "fornet/all/1.0": ("ImageNet", "o"),
        "fornet/all/cos": ("ForNet", "s"),
    }

    ncols = 3
    nrows = math.ceil(len(architectures) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 6 * nrows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, arch in zip(axes, architectures):
        arch_df = df[df["architecture"] == arch]
        for patch in sorted(arch_df["patch_size"].unique(), key=int):
            patch_df = arch_df[arch_df["patch_size"] == patch]
            color = patch_colors.get(patch, "black")
            for dataset_name, (dataset_label, marker) in datasets.items():
                current_df = patch_df[patch_df["train_dataset_name"] == dataset_name]
                if len(current_df) < 2:
                    continue
                current_df = current_df.sort_values("train_dataset_fraction")
                D = current_df["train_dataset_fraction"].to_numpy(float)
                L = current_df["min_val_loss"].to_numpy(float)
                coeffs, fit = calculate_fit(L, D)
                alpha, a = coeffs[0], np.exp(coeffs[1])
                pred = coeffs[0] * np.log(D) + coeffs[1]
                r2 = 1 - np.sum((np.log(L) - pred) ** 2) / np.sum((np.log(L) - np.mean(np.log(L))) ** 2)
                ax.scatter(
                    D, L, s=70, marker=marker, color=color,
                    edgecolor="white", linewidth=0.8, zorder=3
                )
                ax.loglog(
                    D, fit, "--", color=color, lw=1.5,
                    label=rf"P{patch} {dataset_label}: $L={a:.2e}D^{{{alpha:.2f}}}$ ($R^2={r2:.3f}$)"
                )
        ax.set_title(arch, fontsize=13, fontweight="bold")
        ax.set_xlabel("Dataset fraction $D$")
        ax.grid(True, which="major", ls="--", alpha=0.35)
        ax.grid(True, which="minor", ls=":", alpha=0.15)
        show_exact_values(ax, [0.10, 0.25, 0.50, 1.00], "x")
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], marker="o", color="black", ls="", markersize=7),
            Line2D([], [], marker="s", color="black", ls="", markersize=7),
        ]
        labels += ["ImageNet", "ForNet"]
        ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.95)
    for ax in axes[len(architectures):]:
        fig.delaxes(ax)
    for ax in axes[::ncols]:
        if ax in fig.axes:
            ax.set_ylabel("Validation loss $L$")
    fig.suptitle("Dataset Scaling Laws", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "dataset_scaling_comparison.pdf")
    fig.savefig(IMG_OUTPUT_DIR / "dataset_scaling_comparison.png")
    plt.show()



def plot_fg_bg_heatmaps(df:DataFrame):
    df = df[(df["train_dataset_name"] == "fornet/all/cos")].copy()
    models = sorted(df["model_name"].unique())
    fg_order = [1.00, 0.50, 0.25, 0.10]
    bg_order = [0.10, 0.25, 0.50, 1.00]
    n_models = len(models)
    ncols = min(3, n_models)
    nrows = math.ceil(n_models / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 7*nrows))
    axes = np.array(axes).reshape(-1)
    vmin = df["max_val_acc1"].min() * 100
    vmax = df["max_val_acc1"].max() * 100
    for ax, model in zip(axes, models):
        model_df = df[df["model_name"] == model]
        model_df["fg_fraction"] = [float((int(item.split("-")[1])-int(item.split("-")[0]))/100) for item in model_df["fg_range"]]
        model_df["bg_fraction"] = [float((int(item.split("-")[1])-int(item.split("-")[0]))/100) for item in model_df["bg_range"]]
        heatmap = np.zeros((4, 4))
        for i, fg in enumerate(fg_order):
            for j, bg in enumerate(bg_order):
                row = model_df[
                    (model_df["fg_fraction"] == fg) &
                    (model_df["bg_fraction"] == bg)
                ]
                if len(row) == 1:
                    heatmap[i, j] = row["max_val_acc1"].iloc[0] * 100
                else:
                    heatmap[i, j] = np.nan
        im = ax.imshow(
            heatmap,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            aspect="auto"
        )
        ax.grid(False)

        for i in range(4):
            for j in range(4):
                value = heatmap[i, j]
                if not np.isnan(value):
                    ax.text(j,i,f"{value:.2f}",ha="center",va="center",color="white",fontsize=9,fontweight="bold")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(["0.10", "0.25", "0.50", "1.00"])
        ax.set_yticklabels(["1.00", "0.50", "0.25", "0.10"])
        ax.set_xlabel("Background Fraction")
        ax.set_ylabel("Foreground Fraction")
        ax.set_title(model)
    for ax in axes[n_models:]:
        ax.remove()
    cbar = fig.colorbar(im,ax=fig.axes,shrink=0.85)
    cbar.set_label("Top-1 Accuracy (%)")
    fig.subplots_adjust(right=0.75, wspace=0.3, hspace=0.3)
    fig.suptitle("Foreground vs Background Heatmap",fontsize=14)
    fig.savefig(PDF_OUTPUTS_DIR /"fg_bg_heatmaps.pdf")
    fig.savefig(IMG_OUTPUT_DIR /"fg_bg_heatmaps.png")
    plt.show()



def plot_fornet_vs_imagenet_delta_gain(df: DataFrame) -> None:
    data = df.copy()
    data["bg_range"] = data["bg_range"].fillna("null").astype(str).str.lower()
    imagenet = data[
        (data["train_dataset_name"] == "fornet/all/1.0") &
        (data["bg_range"] == "null")
    ]
    fornet = data[
        (data["train_dataset_name"] == "fornet/all/cos")
    ]
    if imagenet.empty or fornet.empty:
        raise ValueError("Missing ImageNet or ForNet runs.")
    models = sorted(set(imagenet["model_name"]).intersection(fornet["model_name"]))
    if not models:
        raise ValueError("No matching models found.")
    # Natural ordering: 0-10, 0-25, 0-50, 0-100
    bg_ranges = sorted(
        [bg for bg in fornet["bg_range"].unique() if bg != "null"],
        key=lambda x: int(x.split("-")[1])
    )
    fractions = [0.1, 0.25, 0.5, 1.0]
    # Global y-limits
    all_deltas = []
    for model in models:
        img = imagenet[
            imagenet.model_name == model
        ].set_index("train_dataset_fraction")["max_val_acc1"]
        fn_model = fornet[fornet.model_name == model]
        for bg in bg_ranges:
            fn = fn_model[
                fn_model.bg_range == bg
            ].set_index("train_dataset_fraction")["max_val_acc1"]
            all_deltas += [
                fn[f] - img[f]
                for f in fractions
                if f in img and f in fn
            ]
    if not all_deltas:
        raise ValueError("No valid delta values found.")
    pad = max(0.02, 0.1 * (max(all_deltas) - min(all_deltas)))
    ymin = min(0, min(all_deltas)) - pad
    ymax = max(all_deltas) + pad
    ncols = 3
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    num_bgs = len(bg_ranges)
    # Publication-friendly colors
    colors = sns.color_palette("colorblind", n_colors=num_bgs)
    group_width = 0.8
    bar_width = group_width / num_bgs
    for ax, model in zip(axes, models):
        img = imagenet[imagenet.model_name == model].set_index("train_dataset_fraction")["max_val_acc1"]
        fn_model = fornet[fornet.model_name == model]
        valid = []
        for f in fractions:
            if f in img:
                has_fn = False
                for bg in bg_ranges:
                    fn = fn_model[fn_model.bg_range == bg].set_index("train_dataset_fraction")["max_val_acc1"]
                    if f in fn:
                        has_fn = True
                        break
                if has_fn:
                    valid.append(f)
        x = np.arange(len(valid))
        for i, bg in enumerate(bg_ranges):
            fn = fn_model[fn_model.bg_range == bg].set_index("train_dataset_fraction")["max_val_acc1"]
            deltas_valid = []
            bar_x_valid = []
            for j, f in enumerate(valid):
                if f in fn:
                    deltas_valid.append(fn[f] - img[f])
                    offset = (i - num_bgs / 2 + 0.5) * bar_width
                    bar_x_valid.append(x[j] + offset)
            if deltas_valid:
                bars = ax.bar(
                    bar_x_valid,
                    deltas_valid,
                    width=bar_width,
                    label=f"{bg}",
                    color=colors[i],
                    edgecolor="0.3",
                    linewidth=0.8,
                )
                for bar, d in zip(bars, deltas_valid):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        d + 0.002 if d >= 0 else d - 0.002,
                        f"{d * 100:+.1f}%",
                        ha="center",
                        va="bottom" if d >= 0 else "top",
                        fontsize=7,
                        rotation=90 if num_bgs > 3 else 0
                    )
        ax.axhline(0,color="black",ls="--",lw=1.2)
        ax.set_title(model)
        ax.set_xlabel("Training dataset fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(valid)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y",ls=":",alpha=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
        if len(valid) > 0:
            ax.legend(
                title="bg_range",
                fontsize="small",
                title_fontsize="small"
            )
    for ax in axes[len(models):]:
        fig.delaxes(ax)
    for ax in axes[::ncols]:
        if ax in fig.axes:
            ax.set_ylabel("Δacc1 [ForNet − ImageNet]")
    fig.tight_layout()
    fig.savefig(
        PDF_OUTPUTS_DIR / "fornet_vs_imagenet_delta_gain.pdf",
        dpi=150
    )
    fig.savefig(
        IMG_OUTPUT_DIR / "fornet_vs_imagenet_delta_gain.png",
        dpi=600
    )
    plt.show()



def plot_model_scaling_comparison(df: DataFrame):
    set_style()
    # x-axis is parameter count N (S/16 and S/32 share N=22M, B/16 and B/28
    # share N=86M) so a proper L ~ a*N^b law can be fit; fits use the true N,
    # but points sharing an N are nudged apart (N_plot) purely for display so
    # markers/labels don't stack, and x-ticks list every model at that N.

    model_params = EXPERIMENT_CONSTANTS["model_parameters"]  # "vit-ti/16" -> N, lowercase keys
    fractions = [0.10, 0.25, 0.50, 1.00]
    models = [m for m in _MODEL_SIZE_ORDER if m in df["model_name"].unique()]
    if not models:
        raise ValueError("No known models (from _MODEL_SIZE_ORDER) found in database.")
    param_count = {m: model_params[m.lower()] for m in models}

    # Group models sharing a parameter count for tick labels, e.g. 22M -> "S/16, S/32"
    models_at_n = {}
    for m, n in param_count.items():
        models_at_n.setdefault(n, []).append(m.split("ViT-")[-1])

    # Nudge models sharing an N apart horizontally (display only) so their
    # markers/labels don't overlap.
    dodge = {}
    for n, group_shorts in models_at_n.items():
        span = 0.16
        offsets = [0.0] if len(group_shorts) == 1 else np.linspace(-span / 2, span / 2, len(group_shorts))
        for short, off in zip(group_shorts, offsets):
            dodge[f"ViT-{short}"] = off
    n_plot = {m: param_count[m] * (1 + dodge[m]) for m in models}

    datasets = {
        "fornet/all/1.0": ("ImageNet", "#1f77b4", "o"),
        "fornet/all/cos": ("ForNet", "#d62728", "s"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    axes = axes.ravel()
    y_bounds = []

    for ax, frac in zip(axes, fractions):
        rows = []
        for model in models:
            img_row = df[
                (df["model_name"] == model) &
                (df["train_dataset_name"] == "fornet/all/1.0") &
                (df["train_dataset_fraction"] == frac)
            ]
            if len(img_row):
                loss = img_row["min_val_loss"].iloc[0]
                rows.append({"model": model, "N": param_count[model], "N_plot": n_plot[model],
                             "dataset_name": "fornet/all/1.0", "loss": loss, "lo": 0.0, "hi": 0.0})

            fn_rows = df[
                (df["model_name"] == model) &
                (df["train_dataset_name"] == "fornet/all/cos") &
                (df["train_dataset_fraction"] == frac)
            ]
            if len(fn_rows):
                vals = fn_rows["min_val_loss"].to_numpy(float)
                mean = vals.mean()
                rows.append({"model": model, "N": param_count[model], "N_plot": n_plot[model],
                             "dataset_name": "fornet/all/cos", "loss": mean,
                             "lo": mean - vals.min(), "hi": vals.max() - mean})
        frac_df = pd.DataFrame(rows)
        y_bounds += (frac_df["loss"] - frac_df["lo"]).tolist()
        y_bounds += (frac_df["loss"] + frac_df["hi"]).tolist()

        for dataset_name, (label, color, marker) in datasets.items():
            sub = frac_df[frac_df["dataset_name"] == dataset_name]
            if sub.empty:
                continue
            N = sub["N"].to_numpy(float)
            N_plot = sub["N_plot"].to_numpy(float)
            L = sub["loss"].to_numpy(float)
            ax.errorbar(
                N_plot, L, yerr=[sub["lo"].to_numpy(float), sub["hi"].to_numpy(float)],
                fmt=marker, color=color, markersize=8, markeredgecolor="white",
                markeredgewidth=0.8, capsize=4, zorder=3, ls="", label=label,
            )
            if len(N) >= 2:
                coeffs, _ = calculate_fit(L, N)
                alpha, a = coeffs[0], np.exp(coeffs[1])
                pred = coeffs[0] * np.log(N) + coeffs[1]
                r2 = 1 - np.sum((np.log(L) - pred) ** 2) / np.sum((np.log(L) - np.mean(np.log(L))) ** 2)
                n_smooth = np.linspace(N.min(), N.max(), 100)
                ax.plot(
                    n_smooth, a * n_smooth ** alpha, "--", color=color, lw=1.5,
                    label=rf"{label} Fit: $L={a:.2f}N^{{{alpha:.2f}}}$ ($R^2={r2:.3f}$)"
                )

        # One label per model (above its highest point) since N_plot already
        # separates models that share a true parameter count.
        for model, g in frac_df.groupby("model"):
            ax.annotate(
                model.split("ViT-")[-1], xy=(g["N_plot"].iloc[0], (g["loss"] + g["hi"]).max()),
                xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=8, color="black", fontweight="bold", clip_on=False,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        n_ticks = sorted(models_at_n)
        ax.xaxis.set_major_locator(ticker.FixedLocator(n_ticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x / 1e6:.1f}M\n({', '.join(models_at_n.get(x, []))})"
        ))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(axis="x", labelsize=8)
        ax.set_title(f"Dataset fraction = {frac:.2f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Parameter count $N$")
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.legend(fontsize=8, loc="best", framealpha=0.95)

    axes[0].set_ylim(min(y_bounds) * 0.85, max(y_bounds) * 1.35)
    for ax in axes[::2]:
        ax.set_ylabel("Validation loss $L$ (min)")
    fig.suptitle("Model Scaling: Validation Loss vs. Parameter Count", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "model_scaling_comparison.pdf")
    fig.savefig(IMG_OUTPUT_DIR / "model_scaling_comparison.png")
    plt.show()



def plot_flops_scaling_comparison(df: DataFrame):
    set_style()

    df = df[df["bg_range"].isin(["0-100", None])].copy()
    df["architecture"] = df["model_name"].str.split("/").str[0]
    df["patch_size"] = df["model_name"].str.split("/").str[1]

    architectures = sorted(df["architecture"].unique())

    patch_colors = {"16": "#1f77b4", "28": "#2ca02c", "32": "#d62728"}
    datasets = {
        "fornet/all/1.0": ("ImageNet", "o"),
        "fornet/all/cos": ("ForNet", "s"),
    }

    ncols = 3
    nrows = math.ceil(len(architectures) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(8 * ncols, 6 * nrows),
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, arch in zip(axes, architectures):
        arch_df = df[df["architecture"] == arch]
        for patch in sorted(arch_df["patch_size"].unique(), key=int):
            patch_df = arch_df[arch_df["patch_size"] == patch]
            color = patch_colors.get(patch, "black")
            for dataset_name, (dataset_label, marker) in datasets.items():
                current_df = patch_df[patch_df["train_dataset_name"] == dataset_name]
                if len(current_df) < 2:
                    continue
                current_df = current_df.sort_values("total_flops")
                C = current_df["total_flops"].to_numpy(float)
                L = current_df["min_val_loss"].to_numpy(float)
                coeffs, fit = calculate_fit(L, C)
                alpha, a = coeffs[0], np.exp(coeffs[1])
                pred = coeffs[0] * np.log(C) + coeffs[1]
                r2 = 1 - np.sum((np.log(L) - pred) ** 2) / np.sum((np.log(L) - np.mean(np.log(L))) ** 2)
                ax.scatter(
                    C, L, s=70, marker=marker, color=color,
                    edgecolor="white", linewidth=0.8, zorder=3
                )
                ax.loglog(
                    C, fit, "--", color=color, lw=1.5,
                    label=rf"P{patch} {dataset_label}: $L={a:.2e}C^{{{alpha:.2f}}}$ ($R^2={r2:.3f}$)"
                )
        ax.set_title(arch, fontsize=13, fontweight="bold")
        ax.set_xlabel("Training compute (FLOPs) $C$")
        ax.grid(True, which="major", ls="--", alpha=0.35)
        ax.grid(True, which="minor", ls=":", alpha=0.15)
        # Auto log ticks can collapse to <4 labels when a single patch size's
        # FLOPs span less than a decade, so pin ticks to the actual FLOPs
        # values present (>=4, since every model has 4 dataset fractions).
        flop_ticks = sorted(float(v) for v in arch_df["total_flops"].unique())
        ax.xaxis.set_major_locator(ticker.FixedLocator(flop_ticks))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x / 1e18:.2f}e18'))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], marker="o", color="black", ls="", markersize=7),
            Line2D([], [], marker="s", color="black", ls="", markersize=7),
        ]
        labels += ["ImageNet", "ForNet"]
        ax.legend(handles, labels, fontsize=8, loc="best", framealpha=0.95)
    for ax in axes[len(architectures):]:
        fig.delaxes(ax)
    for ax in axes[::ncols]:
        if ax in fig.axes:
            ax.set_ylabel("Validation loss $L$")
    fig.suptitle("Compute (FLOPs) Scaling Laws", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "flops_scaling_comparison.pdf")
    fig.savefig(IMG_OUTPUT_DIR / "flops_scaling_comparison.png")
    plt.show()



def plot_extreme_bg_threshold(df: DataFrame, metric: str = "max_val_acc1"):
    """Line plot of `metric` vs bg_fraction (log-x) from the extreme
    background-pool ablation runs. One subplot per (model, fg_range)
    combination actually present in df, one line per mixing schedule
    (train_dataset_name). No hardcoded model list or run count - scales
    to however many models/fg_ranges/schedules are in the dataframe.
    Unknown dataset_name values still plot (fallback marker/color),
    they just won't get a pretty label from _DATASET_LABELS.
    """
    set_style()
    df = df.copy()
    if metric not in ("max_val_acc1", "min_val_loss"):
        raise ValueError(f"Unsupported metric '{metric}'. Use 'max_val_acc1' or 'min_val_loss'.")
    if "bg_fraction" not in df.columns:
        raise ValueError("'bg_fraction' column not found in dataframe.")

    groups = sorted(
        df[["model_name", "fg_range"]].drop_duplicates().itertuples(index=False, name=None)
    )
    n = len(groups)
    if n == 0:
        raise ValueError("No (model_name, fg_range) groups found in dataframe.")

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows), squeeze=False)
    axes = axes.ravel()
    y_label = "Top-1 Validation Accuracy" if metric == "max_val_acc1" else "Validation Loss"

    for ax, (model, fg_range) in zip(axes, groups):
        sub = df[(df["model_name"] == model) & (df["fg_range"] == fg_range)]
        for dataset_name in sorted(sub["train_dataset_name"].unique()):
            label, marker, color = _DATASET_LABELS.get(dataset_name, (dataset_name, "^", None))
            line_df = sub[sub["train_dataset_name"] == dataset_name].sort_values("bg_fraction")
            if len(line_df) < 2:
                # Not enough points to draw a line - still show them as markers.
                ax.scatter(line_df["bg_fraction"], line_df[metric], marker=marker,
                           color=color, s=60, label=label, zorder=3)
                continue
            ax.plot(line_df["bg_fraction"], line_df[metric], marker=marker, color=color,
                    markersize=7, linewidth=1.8, label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("Background fraction (log scale)")
        ax.set_ylabel(y_label)
        ax.set_title(f"{model} | fg_range={fg_range}", fontsize=12, fontweight="bold")
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.grid(True, which="minor", ls=":", alpha=0.15)
        ax.legend(fontsize=9, loc="best", framealpha=0.95)

    for ax in axes[n:]:
        fig.delaxes(ax)

    fig.suptitle("Extreme Background Pool Ablation", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"extreme_bg_threshold_{metric}.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"extreme_bg_threshold_{metric}.png")
    plt.show()



def plot_delta_gain_heatmap(df: DataFrame, metric: str = "max_val_acc1", bg_range: str = "0-100"):
    """Single heatmap of (ForNet - ImageNet) delta across all models and
    dataset fractions present in df, using a fixed bg_range for the
    ForNet side. Rows ordered by parameter_count when available.
    Diverging RdBu colormap centered at 0 makes any negative cell
    (ForNet worse than ImageNet) immediately visible.
    """
    set_style()
    if metric not in ("max_val_acc1", "min_val_loss"):
        raise ValueError(f"Unsupported metric '{metric}'. Use 'max_val_acc1' or 'min_val_loss'.")

    imagenet = df[df["train_dataset_name"] == "fornet/all/1.0"]
    fornet = df[(df["train_dataset_name"] == "fornet/all/cos") & (df["bg_range"] == bg_range)]
    if imagenet.empty or fornet.empty:
        raise ValueError(f"Missing ImageNet or ForNet(bg_range={bg_range}) runs.")

    merged = imagenet.merge(fornet, on=["model_name", "fg_range"], suffixes=("_in", "_fn"))
    if merged.empty:
        raise ValueError("No matching (model, fg_range) pairs between ImageNet and ForNet runs.")

    if metric == "max_val_acc1":
        merged["delta"] = merged["max_val_acc1_fn"] - merged["max_val_acc1_in"]
        cbar_label = "Delta Top-1 Accuracy (ForNet - ImageNet)"
        fmt = lambda v: f"{v*100:+.2f}"
    else:
        # Positive here still means "ForNet better" (i.e. lower loss).
        merged["delta"] = merged["min_val_loss_in"] - merged["min_val_loss_fn"]
        cbar_label = "Delta Val Loss (ImageNet - ForNet, positive = ForNet better)"
        fmt = lambda v: f"{v:+.3f}"

    merged["train_dataset_fraction"] = merged["train_dataset_fraction_in"]

    if "parameter_count_in" in merged.columns:
        order_key = merged.groupby("model_name")["parameter_count_in"].first().sort_values()
        models = order_key.index.tolist()
    else:
        models = sorted(merged["model_name"].unique())

    fractions = sorted(merged["train_dataset_fraction"].unique())

    grid = np.full((len(models), len(fractions)), np.nan)
    for i, m in enumerate(models):
        for j, f in enumerate(fractions):
            cell = merged[(merged["model_name"] == m) & (merged["train_dataset_fraction"] == f)]
            if len(cell) == 1:
                grid[i, j] = cell["delta"].iloc[0]
            elif len(cell) > 1:
                raise ValueError(f"Multiple rows for model={m}, fraction={f} - expected exactly one.")

    vmax = np.nanmax(np.abs(grid))
    fig, ax = plt.subplots(figsize=(1.8 * len(fractions) + 2, 0.9 * len(models) + 2))
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False)

    for i in range(len(models)):
        for j in range(len(fractions)):
            value = grid[i, j]
            if not np.isnan(value):
                text_color = "white" if abs(value) / vmax > 0.55 else "black"
                ax.text(j, i, fmt(value), ha="center", va="center", fontsize=9,
                        fontweight="bold", color=text_color)

    ax.set_xticks(range(len(fractions)))
    ax.set_xticklabels([f"{f:.2f}" for f in fractions])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Training dataset fraction")
    ax.set_ylabel("Model")
    ax.set_title(f"ForNet vs ImageNet Gain (bg_range={bg_range})", fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"delta_gain_heatmap_{metric}.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"delta_gain_heatmap_{metric}.png")
    plt.show()



def plot_combined_scaling_scatter(df: DataFrame, bg_range: str = "0-100"):
    """Overview scatter combining dataset-size, model-size, and compute
    (FLOPs) scaling into one view. NOT a fitted scaling law - points are
    the raw logged values, connected by a plain line in the order they
    were trained (increasing compute), so the shape of each model's
    trajectory is visible without implying any fitted trend.

    One subplot per dataset type (ImageNet | ForNet) instead of one
    combined axes, since 5 models x 4 dataset fractions overlaid together
    became unreadable. Color encodes model (ordered by parameter_count,
    sampled from viridis so the gradient itself signals "small to large"),
    marker size encodes dataset fraction. Both panels share x/y limits -
    total_flops is identical between ImageNet and ForNet for matched
    (model, fg_range) pairs, so the two panels are directly comparable
    side by side.

    Scales to however many models/fractions are present in df - nothing
    is hardcoded. Not intended as the thesis's main figure; built as a
    fast, complete overview for status updates.
    """
    set_style()
    required_cols = {"model_name", "train_dataset_name", "train_dataset_fraction",
                      "total_flops", "min_val_loss", "bg_range", "parameter_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    imagenet = df[df["train_dataset_name"] == "fornet/all/1.0"].copy()
    fornet = df[(df["train_dataset_name"] == "fornet/all/cos") & (df["bg_range"] == bg_range)].copy()
    if imagenet.empty or fornet.empty:
        raise ValueError(f"Missing ImageNet or ForNet(bg_range={bg_range}) runs.")
    models = sorted(
        set(imagenet["model_name"]) | set(fornet["model_name"]),
        key=lambda m: df.loc[df["model_name"] == m, "parameter_count"].iloc[0]
    )
    # Sample viridis away from its two ends and avoid the compressed
    # yellow-green tail, so adjacent model sizes stay visually distinct.
    cmap = plt.get_cmap("viridis")
    color_map = dict(zip(models, [cmap(x) for x in np.linspace(0.05, 0.90, len(models))]))
    fractions = sorted(df["train_dataset_fraction"].unique())
    size_min, size_max = 70, 340
    def frac_to_size(f):
        lo, hi = min(fractions), max(fractions)
        if hi == lo:
            return (size_min + size_max) / 2
        return size_min + (f - lo) / (hi - lo) * (size_max - size_min)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharex=True, sharey=True)
    panel_data = {"ImageNet (baseline)": (axes[0], imagenet), f"ForNet (bg = {bg_range})": (axes[1], fornet)}
    for label, (ax, sub_df) in panel_data.items():
        for model in models:
            model_df = sub_df[sub_df["model_name"] == model].sort_values("total_flops")
            if model_df.empty:
                continue
            sizes = model_df["train_dataset_fraction"].apply(frac_to_size)
            # Plain connecting line through the real points - NOT a fit.
            ax.plot(model_df["total_flops"], model_df["min_val_loss"],
                    color=color_map[model], linewidth=1.4, alpha=0.55, zorder=2)
            ax.scatter(model_df["total_flops"], model_df["min_val_loss"],
                       s=sizes, color=color_map[model], edgecolor="white",
                       linewidth=1.1, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Training compute — total FLOPs (log scale)")
        ax.grid(True, which="major", ls="--", alpha=0.4)
        ax.grid(True, which="minor", ls=":", alpha=0.15)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e18:.1f}e18"))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    axes[0].set_ylabel("Validation loss $L$ (min, log scale)")
    fig.suptitle("Scaling Overview: Compute × Dataset Size × Model Size",
                 fontsize=16, fontweight="bold")
    # Reserve the right ~15% of the canvas for the two legends, so they stay
    # inside the figure (visible in plt.show(), not just in the bbox_inches
    # ="tight" exports) instead of hanging off the axes area.
    fig.tight_layout(rect=(0, 0, 0.845, 0.95))
    # Legend 1: colour = model, ordered small -> large by parameter_count,
    # with the measured parameter count spelled out so the colour gradient
    # is readable as an actual size axis.
    model_labels = []
    for m in models:
        params = df.loc[df["model_name"] == m, "parameter_count"].iloc[0]
        model_labels.append(f"{m}  ({params / 1e6:.1f}M)")
    model_handles = [Line2D([], [], marker="o", color=color_map[m], linestyle="",
                            markersize=9, markeredgecolor="white") for m in models]
    model_legend = fig.legend(model_handles, model_labels, title="Model (parameters)",
                              loc="upper left", bbox_to_anchor=(0.855, 0.92),
                              fontsize=10, title_fontsize=11, frameon=True,
                              labelspacing=0.8, borderpad=0.8)
    fig.add_artist(model_legend)
    # Legend 2: marker size = dataset fraction. markersize is a diameter in
    # points while scatter's s is an area in pt^2, hence the sqrt; the 0.8
    # factor keeps the largest swatch from dominating the legend box.
    size_labels = []
    for f in fractions:
        n_samples = df.loc[df["train_dataset_fraction"] == f, "train_dataset_size"].iloc[0]
        size_labels.append(f"D = {f:.2f}  ({n_samples / 1e3:,.0f}k imgs)")
    size_handles = [Line2D([], [], marker="o", color="none", linestyle="",
                           markersize=(frac_to_size(f) ** 0.5) * 0.8,
                           markerfacecolor="lightgray", markeredgecolor="black")
                    for f in fractions]
    fig.legend(size_handles, size_labels, title="Dataset fraction",
               loc="upper left", bbox_to_anchor=(0.855, 0.50),
               fontsize=10, title_fontsize=11, frameon=True,
               labelspacing=1.4, borderpad=0.8, handletextpad=1.2)
    fig.savefig(PDF_OUTPUTS_DIR / "combined_scaling_overview.pdf", bbox_inches="tight")
    fig.savefig(IMG_OUTPUT_DIR / "combined_scaling_overview.png", bbox_inches="tight")
    plt.show()



def plot_crossover_epoch_vs_dataset_fraction(df: DataFrame, crossover_metric: str = "val_loss") -> None:
    """Line plot of crossover epoch vs. dataset fraction, one line per model.

    For each (model, fg_range) the crossover epoch is averaged across the
    ForNet runs' bg_range variants (crossover is stored per ForNet run,
    relative to the matched ImageNet run at the same fraction). Points where
    none of the bg_range variants ever cross ImageNet are drawn as an
    explicit red "X" at the top of the plot rather than left as a gap.
    """
    set_style()
    crossover_col = f"crossover_epoch_{crossover_metric}"
    if crossover_col not in df.columns:
        raise ValueError(f"Crossover column '{crossover_col}' not found in database. "
                         f"Available columns: {list(df.columns)}")
    fornet = df[df["train_dataset_name"] == "fornet/all/cos"].copy()
    if fornet.empty:
        raise ValueError("No ForNet runs found in database.")

    fractions = [0.10, 0.25, 0.50, 1.00]
    models = sorted(
        fornet["model_name"].unique(),
        key=lambda m: _MODEL_SIZE_ORDER.index(m) if m in _MODEL_SIZE_ORDER else len(_MODEL_SIZE_ORDER)
    )

    # For each model x fraction, average the crossover epoch across bg_range
    # variants that actually crossed (crossover_col > 0); -1 means that
    # particular bg_range never crossed. A fraction is "never" only if none
    # of its bg_range variants crossed.
    epochs_by_model = {}
    for model in models:
        model_df = fornet[fornet["model_name"] == model]
        values = []
        for frac in fractions:
            frac_df = model_df[model_df["train_dataset_fraction"] == frac]
            valid = frac_df.loc[frac_df[crossover_col] > 0, crossover_col]
            mean = valid.mean() if len(valid) > 0 else np.nan
            normalized_epoch = mean / frac_df.iloc[0]["total_epochs"] if mean is not np.nan else np.nan
            values.append(normalized_epoch)
        epochs_by_model[model] = np.array(values, dtype=float)

    all_valid = np.concatenate([v[~np.isnan(v)] for v in epochs_by_model.values()])
    if all_valid.size == 0:
        raise ValueError(f"No valid crossover epochs found for metric '{crossover_metric}'")
    y_max = all_valid.max()
    y_never = y_max * 1.12  # marker row for "never crossed" points, above all real data

    metric_labels = {
        "val_loss": ("Val Loss", "drops below"),
        "val_acc1": ("Val Top-1 Accuracy", "rises above"),
        "val_acc5": ("Val Top-5 Accuracy", "rises above"),
    }
    metric_title, direction = metric_labels.get(crossover_metric, (crossover_metric, "crosses"))

    # Publication-friendly, colorblind-safe categorical palette, fixed order per model.
    colors = sns.color_palette("colorblind", n_colors=len(models))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, model in enumerate(models):
        y = epochs_by_model[model]
        valid_mask = ~np.isnan(y)
        ax.plot(
            np.array(fractions)[valid_mask], y[valid_mask],
            marker='o', markersize=7, linewidth=2, color=colors[i],
            label=model, zorder=3,
        )
        never_mask = ~valid_mask
        if never_mask.any():
            never_x = np.array(fractions)[never_mask]
            ax.scatter(
                never_x, np.full(never_x.shape, y_never),
                marker='x', s=110, linewidth=2.5, color='#d62728',
                zorder=5,
            )
            for nx in never_x:
                ax.annotate(
                    "never", (nx, y_never), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5, color='#d62728', fontweight="bold",
                )

    ax.set_ylim(top=y_never * 1.12)
    ax.set_xlabel("Dataset fraction $D$")
    ax.set_ylabel(f"Crossover epoch (ForNet {direction} ImageNet)")
    ax.set_title(f"Crossover Speed vs. Dataset Fraction ({metric_title})")
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f:.2f}" for f in fractions])
    ax.grid(True, which='major', ls='--', alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker='x', color='#d62728', ls='', markersize=8, markeredgewidth=2.5))
    labels.append("No crossover (ForNet never beats ImageNet)")
    ax.legend(handles, labels, fontsize=9, loc='best', framealpha=0.95)

    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'crossover_speed_{crossover_metric}.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'crossover_speed_{crossover_metric}.png')
    plt.show()



def plot_crossover_vs_model_size(df: DataFrame, crossover_metric: str = "val_loss", normalize: bool = True):
    """Line plot of crossover epoch vs model parameter count (log-x),
    one line per dataset fraction. Mirrors the never-crossed handling
    from plot_crossover_epoch_vs_dataset_fraction for consistency.
    Reads parameter_count directly from df - scales to however many
    models are present, no external EXPERIMENT_CONSTANTS dependency.
    """
    set_style()
    crossover_col = f"crossover_epoch_{crossover_metric}"
    if crossover_col not in df.columns:
        raise ValueError(f"Crossover column '{crossover_col}' not found in database.")
    if "parameter_count" not in df.columns:
        raise ValueError("'parameter_count' column not found in database.")

    fornet = df[df["train_dataset_name"] == "fornet/all/cos"].copy()
    if fornet.empty:
        raise ValueError("No ForNet runs found in database.")

    fractions = sorted(fornet["train_dataset_fraction"].unique())
    models = sorted(
        fornet["model_name"].unique(),
        key=lambda m: fornet.loc[fornet["model_name"] == m, "parameter_count"].iloc[0]
    )
    param_count = {m: fornet.loc[fornet["model_name"] == m, "parameter_count"].iloc[0] for m in models}

    # Per (model, fraction): average crossover epoch across bg_range variants
    # that actually crossed (value > 0); NaN if none of them ever crossed.
    y_by_fraction = {}
    for frac in fractions:
        values = []
        for m in models:
            sub = fornet[(fornet["model_name"] == m) & (fornet["train_dataset_fraction"] == frac)]
            valid = sub.loc[sub[crossover_col] > 0, crossover_col]
            if len(valid) == 0:
                values.append(np.nan)
                continue
            mean_epoch = valid.mean()
            if normalize:
                mean_epoch = mean_epoch / sub.iloc[0]["total_epochs"]
            values.append(mean_epoch)
        y_by_fraction[frac] = np.array(values, dtype=float)

    all_valid = np.concatenate([v[~np.isnan(v)] for v in y_by_fraction.values()])
    if all_valid.size == 0:
        raise ValueError(f"No valid crossover epochs found for metric '{crossover_metric}'.")
    y_max = all_valid.max()
    y_never = y_max * 1.12

    N = np.array([param_count[m] for m in models], dtype=float)
    colors = sns.color_palette("colorblind", n_colors=len(fractions))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, frac in enumerate(fractions):
        y = y_by_fraction[frac]
        valid_mask = ~np.isnan(y)
        ax.plot(N[valid_mask], y[valid_mask], marker="o", markersize=7, linewidth=2,
                color=colors[i], label=f"D={frac:.2f}", zorder=3)
        never_mask = ~valid_mask
        if never_mask.any():
            ax.scatter(N[never_mask], np.full(never_mask.sum(), y_never), marker="x",
                       s=110, linewidth=2.5, color="#d62728", zorder=5)

    ax.set_xscale("log")
    ax.set_ylim(top=y_never * 1.12)
    ax.set_xlabel("Model parameter count $N$ (log scale)")
    ax.set_ylabel("Normalized crossover epoch" if normalize else "Crossover epoch")
    metric_labels = {"val_loss": "Val Loss", "val_acc1": "Val Top-1 Acc", "val_acc5": "Val Top-5 Acc"}
    ax.set_title(f"Crossover Speed vs. Model Size ({metric_labels.get(crossover_metric, crossover_metric)})")
    ax.xaxis.set_major_locator(ticker.FixedLocator(sorted(N)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", ls="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="x", color="#d62728", ls="", markersize=8, markeredgewidth=2.5))
    labels.append("No crossover (never beats ImageNet)")
    ax.legend(handles, labels, fontsize=9, loc="best", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"crossover_vs_model_size_{crossover_metric}.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"crossover_vs_model_size_{crossover_metric}.png")
    plt.show()



def plot_scaling_law(df:DataFrame):
    datasets = df["train_dataset_name"].unique()
    if len(datasets) != 1:
        raise ValueError(f"Dataframe must contain data of only one dataset | Found {datasets}")
    models = df["model_name"].unique()
    dataset_name = None
    if datasets[0] == "fornet/all/cos":
        dataset_name = "ForNet"
    elif datasets[0] == "fornet/all/1.0":
        dataset_name = "ImageNet"
    else:
        raise ValueError(f"Unknown train_dataset_name in database | train_dataset_name: {datasets[0]}")
    fig, ax = plt.subplots(figsize=(15, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    for i, model in enumerate(models):
        filtered_df = filter_database(input_dataframe=df, model_name=model)
        if datasets[0] == "fornet/all/cos":
            filtered_df = filter_database(input_dataframe=filtered_df, bg_range="0-100")
        assert len(filtered_df) == 4
        filtered_df = filtered_df.sort_values("train_dataset_fraction", ascending=True)
        D = filtered_df["train_dataset_fraction"]
        loss = filtered_df["min_val_loss"]
        line, = ax.loglog(D, loss, 'o-', color=colors[i], markersize=8, label=f"{model} val/loss")
        coeffs, fit = calculate_fit(loss, D)
        '''
        # Power‑law fit
        log_D, log_L = np.log(D), np.log(loss)
        coeffs = np.polyfit(log_D, log_L, 1)
        fit = np.exp(coeffs[1]) * D ** coeffs[0]
        '''
        ax.loglog(D, fit, '--', color=line.get_color(), alpha=0.7,
                  label=f'{model} Fit: $L \\propto D^{{{coeffs[0]:.2f}}}$')
    if datasets[0] == "fornet/all/cos":
        ax = show_exact_values(ax, filter_database(input_dataframe=df, bg_range="0-100")["min_val_loss"], "y")
    else: # "fornet/all/1.0"
        ax = show_exact_values(ax, df["min_val_loss"], "y")
    ax = show_exact_values(ax, [0.10, 0.25, 0.50, 1.00], "x")
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.set_xlabel('Dataset fraction $D$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title(f'{dataset_name} Scaling Law: Loss vs Dataset Size')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"{dataset_name.lower()}_scaling_law.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"{dataset_name.lower()}_scaling_law.png")
    plt.show()



def set_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 11,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def calculate_fit(y_axis_values, x_axis_values):
    log_y_values, log_x_values = np.log(y_axis_values), np.log(x_axis_values)
    coefficients = np.polyfit(log_x_values, log_y_values, 1)
    fit = np.exp(coefficients[1]) * x_axis_values ** coefficients[0]
    return coefficients, fit



def show_exact_values(ax, values, axis:str):
    if axis == "x":
        # X axis: exact dataset sizes
        ax.xaxis.set_major_locator(ticker.FixedLocator(values))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{float(x):,}')
        )
        ax.xaxis.set_minor_locator(ticker.NullLocator())  # Remove any minor ticks (which often carry default labels)
    elif axis == "y":
        # Y axis: exact loss values
        ax.yaxis.set_major_locator(ticker.FixedLocator(values))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'{y:.2f}')
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())  # Remove any minor ticks (which often carry default labels)
    return ax


'''
# Due to time limit studying crossover phenomenon is outside thesis scope 
def plot_crossover_flops_scaling(df: DataFrame, crossover_metric: str = "val_loss"):
    set_style()
    # Crossover epochs are only defined relative to ForNet for ImageNet runs
    crossover_col = f"crossover_epoch_{crossover_metric}"
    if crossover_col not in df.columns:
        raise ValueError(f"Crossover column '{crossover_col}' not found in database. "
                         f"Available columns: {list(df.columns)}")
    # Filter for ImageNet dataset where crossover epoch is valid (> 0)
    filtered_df = df[
        (df["train_dataset_name"] == "fornet/all/1.0") &
        (df[crossover_col] > 0)
    ].copy()
    models = sorted(filtered_df["model_name"].unique())
    n_models = len(models)
    if n_models == 0:
        raise ValueError(f"No runs with valid crossover epochs found in database for metric '{crossover_metric}'")
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6 * n_models, 5),
        squeeze=False
    )
    axes = axes[0]  # Flatten to 1D array of axes
    for i, model in enumerate(models):
        ax = axes[i]
        model_df = filtered_df[filtered_df["model_name"] == model].sort_values("train_dataset_size")
        if model_df.empty:
            continue
        D = model_df["train_dataset_size"].values.astype(float)
        crossover_epochs = model_df[crossover_col].values.astype(float)
        flops_per_epoch = model_df["flops_per_epoch"].values.astype(float)
        # Crossover FLOPs = Crossover Epoch * FLOPs per epoch
        crossover_flops = crossover_epochs * flops_per_epoch
        # Plot data points (markers only)
        line, = ax.loglog(D, crossover_flops, 'o', color=f'C{i}', markersize=8, label='Crossover FLOPs data')
        # Calculate fit
        coeffs, fit = calculate_fit(crossover_flops, D)
        b = coeffs[0]  # exponent
        a = np.exp(coeffs[1])  # prefactor
        # Calculate fit quality R^2 in log space
        log_D = np.log(D)
        log_crossover_flops = np.log(crossover_flops)
        pred = coeffs[0] * log_D + coeffs[1]
        ss_res = np.sum((log_crossover_flops - pred) ** 2)
        ss_tot = np.sum((log_crossover_flops - np.mean(log_crossover_flops)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        # Plot power-law fit line
        ax.loglog(D, fit, '--', color=line.get_color(), alpha=0.7,
                  label=rf'Fit: $C_{{\text{{cross}}}} = {a:.2e} \cdot D^{{{b:.2f}}}$ ($R^2 = {r2:.4f}$)')
        # Configure subplot
        ax.set_title(f'{model}')
        ax.set_xlabel('Training dataset samples $D$')
        ax.grid(True, which='major', ls='--', alpha=0.5)
        ax.legend(loc='best', frameon=True)
        # Set exact x values as ticks
        show_exact_values(ax, D, "x")
        # Set exact y values corresponding to data points as ticks
        ax.yaxis.set_major_locator(ticker.FixedLocator(crossover_flops))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'{y:.2e}')
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())
    axes[0].set_ylabel('Crossover FLOPs $C_{\\text{cross}}$')
    fig.suptitle(f'Crossover FLOPs Scaling Law ({crossover_metric.replace("_", " ").title()})', fontsize=15, y=0.98)
    plt.tight_layout()
    # Save the plots
    pdf_path = PDF_OUTPUTS_DIR / f'crossover_flops_{crossover_metric}_scaling.pdf'
    png_path = IMG_OUTPUT_DIR / f'crossover_flops_{crossover_metric}_scaling.png'
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.show()
'''