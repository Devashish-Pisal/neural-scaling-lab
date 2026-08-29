import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from configs.path_config import IMG_OUTPUT_DIR, PDF_OUTPUTS_DIR
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter



dataset_fractions = np.array([0.10, 0.25, 0.50, 1.00])


dataset_images = np.array([
    127_456,
    318_639,
    637_279,
    1_274_557
])

data = {
    "ForNet": {
        0.6: np.array([2.81810, 2.26825, 1.94193, 1.67465]),
        0.8: np.array([3.01851, 2.28326, 1.93666, 1.67465]),
        1.0: np.array([3.26531, 2.33751, 1.93508, 1.67465]),
    },
    "ImageNet-1K": {
        0.6: np.array([3.66279, 2.75740, 2.15921, 1.78267]),
        0.8: np.array([3.80847, 2.77457, 2.16601, 1.78267]),
        1.0: np.array([4.03020, 2.78585, 2.16630, 1.78267]),
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})



DATA_COLOR = "#1F4E79"   # Dark academic blue
FIT_COLOR = "#A23B3B"    # Muted dark red
GRID_COLOR = "#B0B0B0"


fig, axes = plt.subplots(
    2,
    3,
    figsize=(11.5, 6.8),
    sharex=True,
    sharey=True
)

axes = axes.flatten()


plot_configs = [
    ("ForNet", 0.6),
    ("ForNet", 0.8),
    ("ForNet", 1.0),
    ("ImageNet-1K", 0.6),
    ("ImageNet-1K", 0.8),
    ("ImageNet-1K", 1.0),
]



for ax, (dataset_name, beta) in zip(axes, plot_configs):

    loss = data[dataset_name][beta]

    # --------------------------------------------------------
    # Power-law fit:
    #
    # L = a * N^alpha
    #
    # N     = number of training images
    # alpha = fitted validation-loss scaling exponent
    # beta  = prescribed training-step scaling exponent
    #
    # log(L) = log(a) + alpha * log(N)
    # --------------------------------------------------------

    log_N = np.log(dataset_images)
    log_L = np.log(loss)

    regression = linregress(log_N, log_L)

    alpha = regression.slope
    log_a = regression.intercept
    r_squared = regression.rvalue ** 2

    a = np.exp(log_a)


    N_fit = np.logspace(
        np.log10(dataset_images.min()),
        np.log10(dataset_images.max()),
        200
    )

    L_fit = a * N_fit**alpha

    ax.scatter(
        dataset_images,
        loss,
        s=48,
        color=DATA_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="Validation loss"
    )

    ax.plot(
        N_fit,
        L_fit,
        color=FIT_COLOR,
        linestyle="--",
        linewidth=2.0,
        zorder=2,
        label=(
            rf"$L = {a:.2f}N^{{{alpha:.2f}}}$"
            "\n"
            rf"$R^2={r_squared:.4f}$"
        )
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    if ax in [axes[0], axes[3]]:
        ax.set_ylabel("Validation loss")

    ax.set_title(
        rf"{dataset_name}, $\beta={beta:.1f}$",
        pad=8
    )

    ax.xaxis.set_major_locator(
        FixedLocator(dataset_images)
    )

    ax.xaxis.set_major_formatter(
        FixedFormatter([
            "127k",
            "319k",
            "637k",
            "1.27M"
        ])
    )

    # Disable minor tick labels
    ax.xaxis.set_minor_formatter(
        NullFormatter()
    )

    # Remove scientific notation / offset text
    ax.xaxis.get_offset_text().set_visible(False)

    if ax in axes[3:]:
        ax.set_xlabel("Number of training images")

        ax.tick_params(
            axis="x",
            which="major",
            labelbottom=True,
            labelrotation=0,
            pad=5
        )

    else:
        ax.set_xlabel("")

        ax.tick_params(
            axis="x",
            which="major",
            labelbottom=False
        )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.55,
        color=GRID_COLOR,
        alpha=0.45
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.4,
        color=GRID_COLOR,
        alpha=0.25
    )

    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=0.95,
        edgecolor="#B0B0B0",
        handlelength=2.2
    )


fig.suptitle(
    "Scaling of Validation Loss with Dataset Size",
    fontsize=15,
    y=0.995
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97],
    h_pad=1.5,
    w_pad=1.2
)



plt.savefig(
    PDF_OUTPUTS_DIR / "steps_scaling_exponents_comparision.pdf",
    bbox_inches="tight"
)

plt.savefig(
    IMG_OUTPUT_DIR / "steps_scaling_exponents_comparision.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()


print("\nFitted power-law scaling laws:")
print("=" * 90)

for dataset_name, beta in plot_configs:

    loss = data[dataset_name][beta]

    log_N = np.log(dataset_images)
    log_L = np.log(loss)

    regression = linregress(log_N, log_L)

    alpha = regression.slope
    a = np.exp(regression.intercept)
    r_squared = regression.rvalue ** 2

    print(
        f"{dataset_name:12s} | "
        f"beta = {beta:.1f} | "
        f"L = {a:.4f} N^({alpha:.4f}) | "
        f"R^2 = {r_squared:.4f}"
    )