import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    fig, ax = plt.subplots(figsize=(6, 5))
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


def plot_dataset_size_scaling_comparison(df):
    df = df[df["bg_range"].isin(["0-100", None])].copy()
    df["architecture"] = df["model_name"].str.split("/").str[0]
    df["patch_size"] = df["model_name"].str.split("/").str[1]
    architectures = sorted(df["architecture"].unique())
    patch_colors = {
        "16": "#1f77b4",  # Blue
        "28": "#2ca02c",  # Green
        "32": "#d62728"   # Red
    }
    datasets = {"fornet/all/1.0": ("ImageNet", "o"), "fornet/all/cos": ("ForNet", "s")}
    fig, axes = plt.subplots(
        1,
        len(architectures),
        figsize=(6 * len(architectures), 5),
        sharey=True
    )
    if len(architectures) == 1:
        axes = [axes]
    for ax, arch in zip(axes, architectures):
        arch_df = df[df["architecture"] == arch]
        patch_sizes = sorted(
            arch_df["patch_size"].unique(),
            key=int
        )
        for patch in patch_sizes:
            patch_df = arch_df[arch_df["patch_size"] == patch]
            for dataset_name, (dataset_label, marker) in datasets.items():
                current_df = patch_df[patch_df["train_dataset_name"] == dataset_name]
                if current_df.empty:
                    continue
                current_df = current_df.sort_values("train_dataset_fraction")
                D = current_df["train_dataset_fraction"]
                L = current_df["min_val_loss"]
                coeffs, fit = calculate_fit(L,D)
                alpha = coeffs[0]
                ax.loglog(D, L, marker=marker, linestyle='-', markersize=8,
                    color=patch_colors.get(patch,"black"),
                    label=f'P{patch} {dataset_label} ($\\alpha$={alpha:.2f})'
                )
                ax.loglog(D,fit,'--', alpha=0.6,
                    color=patch_colors.get(patch,"black"),
                )
        show_exact_values(ax,[0.10, 0.25, 0.50, 1.00],"x")
        ax.set_title(f'{arch}')
        ax.set_xlabel('Dataset fraction $D$')
        ax.grid(True,which='major',ls='--',alpha=0.5)
        ax.legend(loc='best',fontsize=9,frameon=True)
    axes[0].set_ylabel('Validation loss $L$')
    fig.suptitle('Scaling Law Comparison',fontsize=16)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR /'dataset_scaling_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR /'dataset_scaling_comparison.png')
    plt.show()


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
    vmin = df["max_val_acc1"].min()
    vmax = df["max_val_acc1"].max()
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
                    heatmap[i, j] = row["max_val_acc1"].iloc[0]
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
                    ax.text(j,i,f"{value:.3f}",ha="center",va="center",color="white",fontsize=9,fontweight="bold")
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
    cbar.set_label("Top-1 Accuracy")
    fig.subplots_adjust(right=0.75, wspace=0.3, hspace=0.3)
    fig.suptitle("Foreground vs Background Heatmap",fontsize=14)
    fig.savefig(PDF_OUTPUTS_DIR /"fg_bg_heatmaps.pdf")
    fig.savefig(IMG_OUTPUT_DIR /"fg_bg_heatmaps.png")
    plt.show()


def plot_fornet_vs_imagenet_delta_gain(df: DataFrame) -> None:
    data = df.copy()
    data['bg_range'] = data['bg_range'].fillna('null').astype(str).str.lower()
    imagenet = data[(data['train_dataset_name'] == 'fornet/all/1.0') &
                    (data['bg_range'] == 'null')].copy()
    fornet_bg100 = data[(data['train_dataset_name'] == 'fornet/all/cos') &
                        (data['bg_range'] == '0-100')].copy()
    if imagenet.empty or fornet_bg100.empty:
        raise ValueError("Missing required runs: ImageNet or ForNet (bg=0-100) not found.")
    models_imagenet = set(imagenet['model_name'].unique())
    models_fornet = set(fornet_bg100['model_name'].unique())
    models = sorted(models_imagenet.intersection(models_fornet))
    if not models:
        raise ValueError("No matching model names between ImageNet and ForNet runs.")
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(10, 5), sharey=True)
    if n_models == 1:
        axes = [axes]
    fractions = sorted([0.1, 0.25, 0.5, 1.0])
    for ax, model in zip(axes, models):
        img_df = imagenet[imagenet['model_name'] == model]
        img_acc = img_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        fornet_df = fornet_bg100[fornet_bg100['model_name'] == model]
        fornet_acc = fornet_df.set_index('train_dataset_fraction')['max_val_acc1'].to_dict()
        deltas = []
        valid_fracs = []
        for frac in fractions:
            if frac in img_acc and frac in fornet_acc:
                delta = fornet_acc[frac] - img_acc[frac]
                deltas.append(delta)
                valid_fracs.append(frac)
            else:
                print(f"Warning: Model {model} missing fraction {frac} in either ImageNet or ForNet.")
        if not deltas:
            print(f"Warning: No valid fractions for model {model}. Skipping.")
            ax.set_visible(False)
            continue
        x_pos = np.arange(len(valid_fracs))
        bar_colors = ['green' if d >= 0 else 'red' for d in deltas]
        bars = ax.bar(x_pos, deltas, width=0.6, color=bar_colors, edgecolor='black')
        for bar, val in zip(bars, deltas):
            if val > 0:
                label = f'+{val:.3f}'
            else:
                label = f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    label, ha='center', va='bottom', fontsize=9,
                    color='darkgreen' if val > 0 else 'darkred')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Training dataset fraction', fontsize=10)
        ax.set_ylabel("Δacc1 (ForNet(bg=1.0) − ImageNet)", fontsize=10)
        ax.set_title(f'{model}', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(f) for f in valid_fracs])
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        y_min = min(0, min(deltas)) - 0.02
        y_max = max(deltas) + 0.02
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(PDF_OUTPUTS_DIR / "fornet_vs_imagenet_delta_gain.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(IMG_OUTPUT_DIR / "fornet_vs_imagenet_delta_gain.png", dpi=150, bbox_inches='tight')
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



def plot_dataset_size_scaling_comparison(df):
    df = df[df["bg_range"].isin(["0-100", None])].copy()
    df["architecture"] = df["model_name"].str.split("/").str[0]
    df["patch_size"] = df["model_name"].str.split("/").str[1]
    architectures = sorted(df["architecture"].unique())
    patch_colors = {
        "16": "#1f77b4",  # Blue
        "28": "#2ca02c",  # Green
        "32": "#d62728"   # Red
    }
    datasets = {
        "fornet/all/1.0": ("ImageNet", "o"),
        "fornet/all/cos": ("ForNet", "s")
    }
    fig, axes = plt.subplots(
        1,
        len(architectures),
        figsize=(6 * len(architectures), 5),
        sharey=True
    )
    if len(architectures) == 1:
        axes = [axes]
    for ax, arch in zip(axes, architectures):
        arch_df = df[df["architecture"] == arch]
        patch_sizes = sorted(
            arch_df["patch_size"].unique(),
            key=int
        )
        for patch in patch_sizes:
            patch_df = arch_df[
                arch_df["patch_size"] == patch
            ]
            for dataset_name, (dataset_label, marker) in datasets.items():
                current_df = patch_df[patch_df["train_dataset_name"] == dataset_name]
                if current_df.empty:
                    continue
                current_df = current_df.sort_values("train_dataset_fraction")
                D = current_df["train_dataset_fraction"]
                L = current_df["min_val_loss"]
                coeffs, fit = calculate_fit(L,D)
                alpha = coeffs[0]
                ax.loglog(
                    D,
                    L,
                    marker=marker,
                    linestyle='-',
                    color=patch_colors.get(
                        patch,
                        "black"
                    ),
                    markersize=8,
                    label=(
                        f'P{patch} '
                        f'{dataset_label} '
                        f'($\\alpha$={alpha:.2f})'
                    )
                )
                ax.loglog(
                    D,
                    fit,
                    '--',
                    color=patch_colors.get(
                        patch,
                        "black"
                    ),
                    alpha=0.6
                )
        show_exact_values(ax,[0.10, 0.25, 0.50, 1.00],"x")
        ax.set_title(f'{arch}')
        ax.set_xlabel('Dataset fraction $D$')
        ax.grid(True,which='major',ls='--',alpha=0.5)
        ax.legend(loc='best',fontsize=9,frameon=True)
    axes[0].set_ylabel('Validation loss $L$')
    fig.suptitle('Scaling Law Comparison',fontsize=16)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR /'dataset_scaling_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR /'dataset_scaling_comparison.png')
    plt.show()