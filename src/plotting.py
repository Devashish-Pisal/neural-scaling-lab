import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from src.data_processing import filter_database
from configs.path_config import IMG_OUTPUT_DIR, PDF_OUTPUTS_DIR
from matplotlib.lines import Line2D

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



def plot_steps_allocation(D, steps, ds_name, ref_exponent=0.8):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(D, steps, 'o-', color='C0', markersize=8, label=f'{ds_name} Measured steps')
    # Reference line with the expected scaling exponent
    ref_steps = max(steps) * (D / max(D)) ** ref_exponent
    ax.loglog(D, ref_steps, '--', color='gray', alpha=0.7,
              label=f'Reference: $S \\propto D^{{{ref_exponent}}}$')
    all_steps_values = pd.concat([steps, ref_steps], ignore_index=True)
    show_exact_values(ax, all_steps_values, "y")
    show_exact_values(ax, D, "x")
    ax.set_xlabel('Dataset size $D$')
    ax.set_ylabel('Training steps $S$')
    ax.set_title(f'{ds_name} Steps $S$ vs {ds_name} Samples $D$')
    ax.legend()
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'{ds_name.lower()}_steps_allocation.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'{ds_name.lower()}_steps_allocation.png')
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


'''
def plot_dataset_size_scaling_comparison(df: DataFrame):
    set_style()
    df = df[df["bg_range"].isin(["0-100", None])].copy()
    df["architecture"] = df["model_name"].str.split("/").str[0]
    df["patch_size"] = df["model_name"].str.split("/").str[1]
    architectures = sorted(df["architecture"].unique())
    patch_colors = {
        "16": "#1f77b4",
        "28": "#2ca02c",
        "32": "#d62728",
    }
    datasets = {
        "fornet/all/1.0": ("ImageNet", "o"),
        "fornet/all/cos": ("ForNet", "s"),
    }
    fig, axes = plt.subplots(
        1,
        len(architectures),
        figsize=(8 * len(architectures), 6),
        sharey=True,
    )
    if len(architectures) == 1:
        axes = [axes]
    for ax, arch in zip(axes, architectures):
        arch_df = df[df["architecture"] == arch]
        patch_sizes = sorted(arch_df["patch_size"].unique(),key=int,)
        for patch in patch_sizes:
            patch_df = arch_df[arch_df["patch_size"] == patch]
            color = patch_colors.get(patch, "black")
            for dataset_name, (dataset_label, marker) in datasets.items():
                current_df = patch_df[patch_df["train_dataset_name"] == dataset_name]
                if len(current_df) < 2:
                    continue
                current_df = current_df.sort_values("train_dataset_fraction")
                D = current_df["train_dataset_fraction"].values.astype(float)
                L = current_df["min_val_loss"].values.astype(float)
                coeffs, fit = calculate_fit(L, D)
                alpha = coeffs[0]
                a = np.exp(coeffs[1])
                # R² in log-space
                log_D = np.log(D)
                log_L = np.log(L)
                pred = coeffs[0] * log_D + coeffs[1]
                ss_res = np.sum((log_L - pred) ** 2)
                ss_tot = np.sum((log_L - np.mean(log_L)) ** 2)
                r2 = 1.0 - ss_res / ss_tot
                # DATA POINTS
                ax.scatter(D,L,s=70,marker=marker,color=color,edgecolor="white",linewidth=0.8,alpha=0.95,zorder=3,)
                # FITTED LAW
                ax.loglog(D,fit,"--",lw=1.5,color=color,alpha=0.85,
                    label=(
                        rf"P{patch} {dataset_label}: "
                        rf"$L={a:.2e}D^{{{alpha:.2f}}}$ "
                        rf"($R^2={r2:.3f}$)"
                    ),
                )
        ax.set_title(arch,fontsize=13,fontweight="bold",)
        ax.set_xlabel("Dataset fraction $D$")
        ax.grid(True,which="major",linestyle="--",alpha=0.35,)
        ax.grid(True,which="minor",linestyle=":",alpha=0.15,)
        show_exact_values(ax,[0.10, 0.25, 0.50, 1.00],"x",)
        handles, labels = ax.get_legend_handles_labels()
        # marker explanations
        handles.extend([
            Line2D([], [],marker="o",linestyle="None",color="black",markersize=7,),
            Line2D([], [],marker="s",linestyle="None",color="black",markersize=7,),
        ])
        labels.extend(["ImageNet","ForNet",])
        ax.legend(handles,labels,fontsize=8,frameon=True,framealpha=0.95,loc="best",handlelength=2.5,)
    axes[0].set_ylabel("Validation loss $L$")
    fig.suptitle("Dataset Scaling Laws",fontsize=16,fontweight="bold",)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "dataset_scaling_comparison.pdf")
    fig.savefig(IMG_OUTPUT_DIR / "dataset_scaling_comparison.png")
    plt.show()
'''

def plot_fg_bg_heatmaps(df:DataFrame):
    df = df[(df["train_dataset_name"] == "fornet/all/cos")].copy()
    models = sorted(df["model_name"].unique())
    fg_order = [1.00, 0.50, 0.25, 0.10]
    bg_order = [1.00, 0.50, 0.25, 0.10]
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
        (data["train_dataset_name"] == "fornet/all/cos") &
        (data["bg_range"] == "0-100")
    ]
    if imagenet.empty or fornet.empty:
        raise ValueError("Missing ImageNet or ForNet runs.")
    models = sorted(
        set(imagenet["model_name"]).intersection(fornet["model_name"])
    )
    if not models:
        raise ValueError("No matching models found.")
    fractions = [0.1, 0.25, 0.5, 1.0]
    # Global y-limits
    all_deltas = []
    for model in models:
        img = imagenet[imagenet.model_name == model].set_index("train_dataset_fraction")["max_val_acc1"]
        fn = fornet[fornet.model_name == model].set_index("train_dataset_fraction")["max_val_acc1"]
        all_deltas += [fn[f] - img[f] for f in fractions if f in img and f in fn]
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
    for ax, model in zip(axes, models):
        img = imagenet[imagenet.model_name == model].set_index("train_dataset_fraction")["max_val_acc1"]
        fn = fornet[fornet.model_name == model].set_index("train_dataset_fraction")["max_val_acc1"]
        valid = [f for f in fractions if f in img and f in fn]
        deltas = [fn[f] - img[f] for f in valid]
        x = np.arange(len(valid))
        bars = ax.bar(
            x,
            deltas,
            color=["green" if d >= 0 else "red" for d in deltas],
            edgecolor="black",
            width=0.6,
        )
        for bar, d in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                d + 0.002 if d >= 0 else d - 0.002,
                f"{d * 100:+.2f}%",
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=9,
            )
        ax.axhline(0, color="black", ls="--", lw=1.2)
        ax.set_title(model)
        ax.set_xlabel("Training dataset fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(valid)
        ax.set_ylim(ymin, ymax)
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    for ax in axes[len(models):]:
        fig.delaxes(ax)
    for ax in axes[::ncols]:
        if ax in fig.axes:
            ax.set_ylabel("Δacc1 [ForNet(bg=1.0) − ImageNet]")
    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "fornet_vs_imagenet_delta_gain.pdf", dpi=150)
    fig.savefig(IMG_OUTPUT_DIR / "fornet_vs_imagenet_delta_gain.png", dpi=150)
    plt.show()



'''
def plot_fornet_vs_imagenet_delta_gain(df: DataFrame) -> None:
    data = df.copy()
    data['bg_range'] = data['bg_range'].fillna('null').astype(str).str.lower()
    imagenet = data[
        (data['train_dataset_name'] == 'fornet/all/1.0') &
        (data['bg_range'] == 'null')
    ].copy()
    fornet_bg100 = data[
        (data['train_dataset_name'] == 'fornet/all/cos') &
        (data['bg_range'] == '0-100')
    ].copy()
    if imagenet.empty or fornet_bg100.empty:
        raise ValueError("Missing required runs: ImageNet or ForNet (bg=0-100) not found.")
    models_imagenet = set(imagenet['model_name'].unique())
    models_fornet = set(fornet_bg100['model_name'].unique())
    models = sorted(models_imagenet.intersection(models_fornet))
    if not models:
        raise ValueError("No matching model names between ImageNet and ForNet runs.")
    fractions = [0.1, 0.25, 0.5, 1.0]

    # Compute global y-axis limits across all models
    all_deltas = []
    for model in models:
        img_df = imagenet[imagenet['model_name'] == model]
        img_acc = img_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        fornet_df = fornet_bg100[fornet_bg100['model_name'] == model]
        fornet_acc = fornet_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        for frac in fractions:
            if frac in img_acc and frac in fornet_acc:
                all_deltas.append(fornet_acc[frac] - img_acc[frac])
    if not all_deltas:
        raise ValueError("No valid delta values found.")
    padding = max(0.02, 0.1 * (max(all_deltas) - min(all_deltas)))
    global_ymin = min(0, min(all_deltas)) - padding
    global_ymax = max(all_deltas) + padding

    n_models = len(models)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(16, 8),
        sharey=True
    )
    if n_models == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        img_df = imagenet[imagenet['model_name'] == model]
        img_acc = img_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        fornet_df = fornet_bg100[fornet_bg100['model_name'] == model]
        fornet_acc = fornet_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        deltas = []
        valid_fracs = []
        for frac in fractions:
            if frac in img_acc and frac in fornet_acc:
                deltas.append(fornet_acc[frac] - img_acc[frac])
                valid_fracs.append(frac)
            else:
                print(
                    f"Warning: Model {model} missing fraction {frac} "
                    "in either ImageNet or ForNet."
                )
        if not deltas:
            print(f"Warning: No valid fractions for model {model}. Skipping.")
            ax.set_visible(False)
            continue
        x_pos = np.arange(len(valid_fracs))
        colors = ["green" if d >= 0 else "red" for d in deltas]
        bars = ax.bar(
            x_pos,
            deltas,
            width=0.6,
            color=colors,
            edgecolor="black"
        )
        for bar, val in zip(bars, deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.002 if val >= 0 else val - 0.002,
                f"{val * 100:+.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
                color="darkgreen" if val >= 0 else "darkred",
            )
        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )
        ax.set_xlabel("Training dataset fraction", fontsize=10)
        ax.set_title(model, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(f) for f in valid_fracs])
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        ax.set_ylim(global_ymin, global_ymax)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
        )
    axes[0].set_ylabel(
        "Δacc1 (ForNet(bg=0-100) − ImageNet)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(
        PDF_OUTPUTS_DIR / "fornet_vs_imagenet_delta_gain.pdf",
        dpi=150,
        bbox_inches="tight",
    )
    plt.savefig(
        IMG_OUTPUT_DIR / "fornet_vs_imagenet_delta_gain.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()
'''

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



def plot_flops_scaling_comparison(flops, imgnet_loss, fornet_loss):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(flops,imgnet_loss, 'o-', color='C3', markersize=8, label='ImageNet')
    ax.loglog(flops,fornet_loss, 's-', color='C4', markersize=8, label='ForNet')
    # Power‑law fit for ImageNet
    log_flops_imgnet, log_L_imgnet = np.log(flops), np.log(imgnet_loss)
    coeffs_imgnet = np.polyfit(log_flops_imgnet, log_L_imgnet, 1)
    fit_imgnet = np.exp(coeffs_imgnet[1]) * flops ** coeffs_imgnet[0]
    ax.loglog(flops, fit_imgnet, '--', color='orange', alpha=0.7,
              label=f'ImageNet Fit: $L \\propto C^{{{coeffs_imgnet[0]:.2f}}}$')
    # Power‑law fit for ForNet
    log_flops_fornet, log_L_fornet = np.log(flops), np.log(fornet_loss)
    coeffs_fornet = np.polyfit(log_flops_fornet, log_L_fornet, 1)
    fit_fornet = np.exp(coeffs_fornet[1]) * flops ** coeffs_fornet[0]
    ax.loglog(flops, fit_fornet, '--', color='green', alpha=0.7,
              label=f'ForNet Fit: $L \\propto C^{{{coeffs_fornet[0]:.2f}}}$')
    all_loss_values =  pd.concat([imgnet_loss, fornet_loss], ignore_index=True)
    ax = show_exact_values(ax, all_loss_values, "y")
    ax.xaxis.set_major_locator(ticker.FixedLocator(flops))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x / 1e18:.1f}e18')
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Training Compute (FLOPs) $C$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title('Compute Scaling Comparison: ImageNet vs ForNet')
    ax.legend()
    ax.grid(True, which='major', ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / 'flops_scaling_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / 'flops_scaling_comparison.png')
    plt.show()


def plot_acc_comparison(D, imgnet_acc, fornet_acc, acc_type:str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(D, imgnet_acc, 'o-', color='C2', markersize=8, label="ImageNet") # Linear plot
    ax.plot(D, fornet_acc, 'd-', color='C3', markersize=8, label="ForNet") # Linear plot
    all_accuracy_values = pd.concat([imgnet_acc, fornet_acc] , ignore_index=True)
    show_exact_values(ax, all_accuracy_values, "y")
    show_exact_values(ax, D, "x")
    ax.set_xlabel('Dataset size $D$')
    ax.set_ylabel(f'{acc_type} Accuracy (%)')
    ax.set_title(f'{acc_type} Accuracy vs Dataset Size')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'{acc_type.lower()}_acc_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'{acc_type.lower()}_acc_comparison.png')
    plt.show()


def plot_compute_efficiency_comparison(flops, imgnet_acc, fornet_acc, acc_type='Top-1'):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(flops, imgnet_acc, 'o-', color='C2', markersize=8, label='ImageNet')
    ax.plot(flops, fornet_acc, 's-', color='C3', markersize=8, label='ForNet')
    all_acc = np.concatenate([imgnet_acc, fornet_acc])
    show_exact_values(ax, all_acc, "y")
    ax.xaxis.set_major_locator(ticker.FixedLocator(flops))
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x/1e18:.1f}e18')
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel('Training compute (FLOPs) $C$')
    ax.set_ylabel(f'{acc_type} Accuracy (%)')
    ax.set_title(f'Compute Efficiency: {acc_type} Accuracy vs FLOPs')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'{acc_type.lower()}_compute_efficiency_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'{acc_type.lower()}_compute_efficiency_comparison.png')
    plt.show()


def plot_sample_efficiency_comparison(D, imgnet_acc, fornet_acc, acc_type='Top-1',
                           target_accuracy=None):
    set_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(D, imgnet_acc, 'o-', color='C2', markersize=8, label='ImageNet')
    ax.plot(D, fornet_acc, 'd-', color='C3', markersize=8, label='ForNet')
    all_acc = np.concatenate([imgnet_acc, fornet_acc])
    show_exact_values(ax, all_acc, "y")
    show_exact_values(ax, D, "x")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{int(x):,}')
    )
    if target_accuracy is not None:
        ax.axhline(y=target_accuracy, color='grey', linestyle=':', alpha=0.7,
                   label=f'{target_accuracy}% target')
        ax.legend()
    ax.set_xlabel('Number of training samples $D$')
    ax.set_ylabel(f'{acc_type} Accuracy (%)')
    ax.set_title(f'Sample Efficiency: {acc_type} Accuracy vs Dataset Size')
    ax.legend()
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'{acc_type.lower()}_sample_efficiency_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'{acc_type.lower()}_sample_efficiency_comparison.png')
    plt.show()


def plot_exponent_scaling_consistency( all_results, csv_config, ds_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {
        0.6: 'C0',
        0.8: 'C1',
        1.0: 'C2'
    }
    for exponent, result in all_results.items():
        if ds_name == "ImageNet":
            data = result["imagenet"]
        else:
            data = result["fornet"]
        D = data[csv_config.real_ds_samples]
        L = data[csv_config.min_val_loss]
        log_D = np.log(D)
        log_L = np.log(L)
        coeffs = np.polyfit(log_D,log_L,1)
        k = -coeffs[0]
        fit = (np.exp(coeffs[1]) *D ** coeffs[0])
        ax.loglog(D,L,'o-',color=colors[exponent],markersize=8,
            label=(
                fr'$\alpha={exponent}$, '
                fr'$L \propto D^{{-{k:.2f}}}$'
            )
        )
        ax.loglog(D, fit,'--', color=colors[exponent], alpha=0.5)
    show_exact_values(ax,D,"x")
    ax.set_xlabel('Dataset size $D$')
    ax.set_ylabel('Best validation loss $L$' )
    ax.set_title(f'{ds_name}: exponent comparison' )
    ax.legend()
    ax.grid(True,which='major',ls='--',alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR /f'{ds_name.lower()}_exponent_scaling.pdf')
    fig.savefig(IMG_OUTPUT_DIR /f'{ds_name.lower()}_exponent_scaling.png')
    plt.show()



def plot_exponent_fit_quality(all_results,csv_config):
    exponents = []
    r2_scores = []
    for exponent, result in all_results.items():
        dataset_r2_scores = []
        for ds_name in ["imagenet", "fornet"]:
            data = result[ds_name]
            D = data[csv_config.real_ds_samples]
            L = data[csv_config.min_val_loss]
            log_D = np.log(D)
            log_L = np.log(L)
            coeffs = np.polyfit(log_D,log_L,1)
            pred = (coeffs[0] * log_D +coeffs[1])
            ss_res = np.sum((log_L - pred) ** 2)
            ss_tot = np.sum((log_L - np.mean(log_L)) ** 2)
            r2 = (1 -(ss_res / ss_tot))
            dataset_r2_scores.append( r2)
        exponents.append(exponent)
        r2_scores.append(np.mean(dataset_r2_scores))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(exponents,r2_scores,width=0.12)
    min_r2 = min(r2_scores)
    zoom_margin = 0.01
    ax.set_ylim(min_r2 - zoom_margin,1.0)
    for bar, r2 in zip(bars,r2_scores):
        ax.text(bar.get_x() +bar.get_width() / 2,r2 + 0.001,f'{r2:.4f}',ha='center',va='bottom',fontsize=11)
    ax.set_xlabel('Scaling exponent $\\alpha$')
    ax.set_ylabel('Mean $R^2$')
    ax.set_title('Power-law fit quality')
    ax.grid(True,axis='y',ls='--',alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR /'exponent_fit_quality.pdf')
    fig.savefig(IMG_OUTPUT_DIR /'exponent_fit_quality.png')
    plt.show()


def plot_exponent_compute_efficiency(all_results,csv_config,ds_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {0.6: 'C0',0.8: 'C1',1.0: 'C2'}
    for exponent, result in all_results.items():
        if ds_name == "ImageNet":
            data = result["imagenet"]
        else:
            data = result["fornet"]
        flops = data[csv_config.real_FLOPs]
        acc = data[csv_config.max_val_acc1]
        ax.plot(flops,acc,'o-',color=colors[exponent],markersize=8,label=fr'$\alpha={exponent}$')
    ax.xaxis.set_major_locator(
        ticker.FixedLocator(flops)
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _:f'{x/1e18:.1f}e18')
    )
    ax.set_xlabel('Training compute (FLOPs) $C$')
    ax.set_ylabel('Top-1 accuracy (%)')
    ax.set_title(f'{ds_name}: compute efficiency')
    ax.legend()
    ax.grid(True,ls='--',alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR /f'{ds_name.lower()}_compute_efficiency_exponent.pdf')
    fig.savefig(IMG_OUTPUT_DIR /f'{ds_name.lower()}_compute_efficiency_exponent.png')
    plt.show()



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

