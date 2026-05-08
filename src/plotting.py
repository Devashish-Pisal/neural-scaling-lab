
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


def plot_steps_allocation(N, steps, ds_name, ref_exponent=0.8):
    """
    Verifies that training steps follow S ∝ N^0.8 .
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(N, steps, 'o-', color='C0', markersize=8, label=f'{ds_name} Measured steps')

    # Reference line with the expected scaling exponent
    ref_steps = max(steps) * (N / max(N)) ** ref_exponent
    ax.loglog(N, ref_steps, '--', color='gray', alpha=0.7,
              label=f'Reference: $S \\propto N^{{{ref_exponent}}}$')
    all_steps_values = pd.concat([steps, ref_steps], ignore_index=True)
    show_exact_values(ax, all_steps_values, "y")
    show_exact_values(ax, N, "x")
    ax.set_xlabel('Dataset size $N$')
    ax.set_ylabel('Training steps $S$')
    ax.set_title(f'{ds_name} Steps $S$ vs {ds_name} Samples $N$')
    ax.legend()
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'{ds_name.lower()}_steps_allocation.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'{ds_name.lower()}_steps_allocation.png')
    plt.show()


def plot_flops_allocation():
    pass


def plot_scaling_law(N, loss, ds_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(N, loss, 's-', color='C1', markersize=8, label='Validation loss')
    # Power‑law fit
    log_N, log_L = np.log(N), np.log(loss)
    coeffs = np.polyfit(log_N, log_L, 1)
    fit = np.exp(coeffs[1]) * N ** coeffs[0]
    ax.loglog(N, fit, '--', color='gray', alpha=0.7,
              label=f'Fit: $L \\propto N^{{{coeffs[0]:.2f}}}$')

    ax = show_exact_values(ax, loss, "y")
    ax = show_exact_values(ax, N, "x")

    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.set_xlabel('Dataset size $N$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title(f'{ds_name} Scaling Law: Loss vs Dataset Size')
    ax.legend()
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f"{ds_name.lower()}_scaling_law.pdf")
    fig.savefig(IMG_OUTPUT_DIR / f"{ds_name.lower()}_scaling_law.png")
    plt.show()


def plot_imagenet_vs_fornet_scaling_comparison(N, loss_imagenet, loss_fornet):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.loglog(N, loss_imagenet, 'o-', color='C3', markersize=8, label='ImageNet')
    ax.loglog(N, loss_fornet, 's-', color='C4', markersize=8, label='ForNet')

    # Power‑law fit for ImageNet
    log_N_imgnet, log_L_imgnet = np.log(N), np.log(loss_imagenet)
    coeffs_imgnet = np.polyfit(log_N_imgnet, log_L_imgnet, 1)
    fit_imgnet = np.exp(coeffs_imgnet[1]) * N ** coeffs_imgnet[0]
    ax.loglog(N, fit_imgnet, '--', color='orange', alpha=0.7,
              label=f'ImageNet Fit: $L \\propto N^{{{coeffs_imgnet[0]:.2f}}}$')
    # Power‑law fit for ForNet
    log_N_fornet, log_L_fornet = np.log(N), np.log(loss_fornet)
    coeffs_fornet = np.polyfit(log_N_fornet, log_L_fornet, 1)
    fit_fornet = np.exp(coeffs_fornet[1]) * N ** coeffs_fornet[0]
    ax.loglog(N, fit_fornet, '--', color='green', alpha=0.7,
              label=f'ForNet Fit: $L \\propto N^{{{coeffs_fornet[0]:.2f}}}$')

    all_loss_values =  pd.concat([loss_imagenet, loss_fornet], ignore_index=True)
    ax = show_exact_values(ax, all_loss_values, "y")
    ax = show_exact_values(ax, N, "x")

    ax.set_xlabel('Dataset size $N$')
    ax.set_ylabel('Validation loss $L$')
    ax.set_title('Scaling Comparison: ImageNet vs ForNet')
    ax.legend()
    ax.grid(True, which='major', ls='--', alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / 'imagenet_vs_fornet_scaling_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / 'imagenet_vs_fornet_scaling_comparison.png')
    plt.show()


def plot_imagenet_vs_fornet_acc_comparison(N, imgnet_acc, fornet_acc, acc_type:str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(N, imgnet_acc, 'o-', color='C2', markersize=8, label="ImageNet") # Linear plot
    ax.plot(N, fornet_acc, 'd-', color='C3', markersize=8, label="ForNet") # Linear plot
    all_accuracy_values = pd.concat([imgnet_acc, fornet_acc] , ignore_index=True)
    show_exact_values(ax, all_accuracy_values, "y")
    show_exact_values(ax, N, "x")

    ax.set_xlabel('Dataset size $N$')
    ax.set_ylabel(f'{acc_type} Accuracy (%)')
    ax.set_title(f'{acc_type} Accuracy vs Dataset Size')
    ax.legend()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / f'imagenet_vs_fornet_{acc_type.lower()}_acc_comparison.pdf')
    fig.savefig(IMG_OUTPUT_DIR / f'imagenet_vs_fornet_{acc_type.lower()}_acc_comparison.png')
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