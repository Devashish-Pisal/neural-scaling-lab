
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from configs.path_config import IMG_OUTPUT_DIR, PDF_OUTPUTS_DIR


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



def plot_scaling_law(D, loss, ds_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(D, loss, 's-', color='C1', markersize=8, label='Validation loss')
    # Power‑law fit
    log_D, log_L = np.log(D), np.log(loss)
    coeffs = np.polyfit(log_D, log_L, 1)
    fit = np.exp(coeffs[1]) * D ** coeffs[0]
    ax.loglog(D, fit, '--', color='gray', alpha=0.7,
              label=f'Fit: $L \\propto D^{{{coeffs[0]:.2f}}}$')
    ax = show_exact_values(ax, loss, "y")
    ax = show_exact_values(ax, D, "x")
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.set_xlabel('Dataset size $D$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title(f'{ds_name} Scaling Law: Loss vs Dataset Size')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"{ds_name.lower()}_scaling_law.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"{ds_name.lower()}_scaling_law.png")
    plt.show()


def plot_dataset_size_scaling_comparison(D, imgnet_loss, fornet_loss):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(D, imgnet_loss, 'o-', color='C3', markersize=8, label='ImageNet')
    ax.loglog(D, fornet_loss, 's-', color='C4', markersize=8, label='ForNet')
    # Power‑law fit for ImageNet
    log_D_imgnet, log_L_imgnet = np.log(D), np.log(imgnet_loss)
    coeffs_imgnet = np.polyfit(log_D_imgnet, log_L_imgnet, 1)
    fit_imgnet = np.exp(coeffs_imgnet[1]) * D ** coeffs_imgnet[0]
    ax.loglog(D, fit_imgnet, '--', color='orange', alpha=0.7,
              label=f'ImageNet Fit: $L \\propto D^{{{coeffs_imgnet[0]:.2f}}}$')
    # Power‑law fit for ForNet
    log_D_fornet, log_L_fornet = np.log(D), np.log(fornet_loss)
    coeffs_fornet = np.polyfit(log_D_fornet, log_L_fornet, 1)
    fit_fornet = np.exp(coeffs_fornet[1]) * D ** coeffs_fornet[0]
    ax.loglog(D, fit_fornet, '--', color='green', alpha=0.7,
              label=f'ForNet Fit: $L \\propto D^{{{coeffs_fornet[0]:.2f}}}$')
    all_loss_values =  pd.concat([imgnet_loss, fornet_loss], ignore_index=True)
    ax = show_exact_values(ax, all_loss_values, "y")
    ax = show_exact_values(ax, D, "x")
    ax.set_xlabel('Dataset size $D$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title('Dataset Scaling Comparison: ImageNet vs ForNet')
    ax.legend()
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / 'dataset_size_scaling_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / 'dataset_size_scaling_comparison.png')
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


def plot_exponent_scaling_consistency(
        all_results,
        csv_config,
        ds_name):
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
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
        D = data[
            csv_config.real_ds_samples
        ]
        L = data[
            csv_config.min_val_loss
        ]
        log_D = np.log(D)
        log_L = np.log(L)
        coeffs = np.polyfit(
            log_D,
            log_L,
            1
        )
        k = -coeffs[0]
        fit = (
            np.exp(coeffs[1]) *
            D ** coeffs[0]
        )
        ax.loglog(
            D,
            L,
            'o-',
            color=colors[exponent],
            markersize=8,
            label=(
                fr'$\alpha={exponent}$, '
                fr'$L \propto D^{{-{k:.2f}}}$'
            )
        )
        ax.loglog(
            D,
            fit,
            '--',
            color=colors[exponent],
            alpha=0.5
        )
    show_exact_values(
        ax,
        D,
        "x"
    )
    ax.set_xlabel(
        'Dataset size $D$'
    )
    ax.set_ylabel(
        'Best validation loss $L$'
    )
    ax.set_title(
        f'{ds_name}: exponent comparison'
    )
    ax.legend()
    ax.grid(
        True,
        which='major',
        ls='--',
        alpha=0.5
    )
    plt.tight_layout()
    fig.savefig(
        PDF_OUTPUTS_DIR /
        f'{ds_name.lower()}_exponent_scaling.pdf'
    )
    fig.savefig(
        IMG_OUTPUT_DIR /
        f'{ds_name.lower()}_exponent_scaling.png'
    )
    plt.show()



def plot_exponent_fit_quality(
        all_results,
        csv_config):
    exponents = []
    r2_scores = []
    for exponent, result in all_results.items():
        dataset_r2_scores = []
        for ds_name in ["imagenet", "fornet"]:
            data = result[ds_name]
            D = data[
                csv_config.real_ds_samples
            ]
            L = data[
                csv_config.min_val_loss
            ]
            log_D = np.log(D)
            log_L = np.log(L)
            coeffs = np.polyfit(
                log_D,
                log_L,
                1
            )
            pred = (
                coeffs[0] * log_D +
                coeffs[1]
            )
            ss_res = np.sum(
                (log_L - pred) ** 2
            )
            ss_tot = np.sum(
                (log_L - np.mean(log_L)) ** 2
            )
            r2 = (
                1 -
                (ss_res / ss_tot)
            )
            dataset_r2_scores.append(
                r2
            )
        exponents.append(
            exponent
        )
        r2_scores.append(
            np.mean(
                dataset_r2_scores
            )
        )
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
    bars = ax.bar(
        exponents,
        r2_scores,
        width=0.12
    )
    min_r2 = min(
        r2_scores
    )
    zoom_margin = 0.01
    ax.set_ylim(
        min_r2 - zoom_margin,
        1.0
    )
    for bar, r2 in zip(
            bars,
            r2_scores):
        ax.text(
            bar.get_x() +
            bar.get_width() / 2,
            r2 + 0.001,
            f'{r2:.4f}',
            ha='center',
            va='bottom',
            fontsize=11
        )
    ax.set_xlabel(
        'Scaling exponent $\\alpha$'
    )
    ax.set_ylabel(
        'Mean $R^2$'
    )
    ax.set_title(
        'Power-law fit quality'
    )
    ax.grid(
        True,
        axis='y',
        ls='--',
        alpha=0.5
    )
    plt.tight_layout()
    fig.savefig(
        PDF_OUTPUTS_DIR /
        'exponent_fit_quality.pdf'
    )
    fig.savefig(
        IMG_OUTPUT_DIR /
        'exponent_fit_quality.png'
    )
    plt.show()


def plot_exponent_compute_efficiency(
        all_results,
        csv_config,
        ds_name):
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
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
        flops = data[
            csv_config.real_FLOPs
        ]
        acc = data[
            csv_config.max_val_acc1
        ]
        ax.plot(
            flops,
            acc,
            'o-',
            color=colors[exponent],
            markersize=8,
            label=fr'$\alpha={exponent}$'
        )
    ax.xaxis.set_major_locator(
        ticker.FixedLocator(
            flops
        )
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _:
            f'{x/1e18:.1f}e18'
        )
    )
    ax.set_xlabel(
        'Training compute (FLOPs) $C$'
    )
    ax.set_ylabel(
        'Top-1 accuracy (%)'
    )
    ax.set_title(
        f'{ds_name}: compute efficiency'
    )
    ax.legend()
    ax.grid(
        True,
        ls='--',
        alpha=0.5
    )
    plt.tight_layout()
    fig.savefig(
        PDF_OUTPUTS_DIR /
        f'{ds_name.lower()}_compute_efficiency_exponent.pdf'
    )
    fig.savefig(
        IMG_OUTPUT_DIR /
        f'{ds_name.lower()}_compute_efficiency_exponent.png'
    )
    plt.show()



def show_exact_values(ax, values, axis:str):
    if axis == "x":
        # X axis: exact dataset sizes
        ax.xaxis.set_major_locator(ticker.FixedLocator(values))
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'{int(x):,}')
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