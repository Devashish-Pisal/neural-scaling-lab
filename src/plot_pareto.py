import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from configs.path_config import IMG_OUTPUT_DIR, PDF_OUTPUTS_DIR
from src.plotting import set_style, _MODEL_SIZE_ORDER

# Double-saturating power law, Sec 2.2 of "Scaling Vision Transformers": E(C) = c + a*(C+d)^-b.
def _double_saturating_power_law(compute, a, b, c, d):
    return c + a * np.power(compute + d, -b)


def _pareto_frontier(compute, error):
    # Lower envelope: only points that set a new best (lowest) error as compute grows survive.
    order = np.argsort(compute)
    c_sorted, e_sorted = compute[order], error[order]
    frontier_c, frontier_e = [], []
    best = np.inf
    for c, e in zip(c_sorted, e_sorted):
        if e < best:
            frontier_c.append(c)
            frontier_e.append(e)
            best = e
    return np.array(frontier_c), np.array(frontier_e)


def _model_color_positions(n):
    # coolwarm's center is near-white, so skip that band to keep every
    # model's color visibly blue- or red-tinted instead of washed out.
    if n <= 1:
        return np.array([0.05])
    dead_lo, dead_hi = 0.40, 0.62
    span = dead_lo + (1.0 - dead_hi)
    positions = []
    for i in range(n):
        target = (i / (n - 1)) * span
        positions.append(target if target <= dead_lo else dead_hi + (target - dead_lo))
    return np.array(positions)


def _fit_scaling_law(compute, error, n_restarts=25, seed=0):
    # Random multi-start: with only ~8-20 frontier points, a single start
    # can land on a poor local optimum; keep the restart with the best R^2.
    bounds = ([1e-6, 0.01, 0.0, 0.0], [100.0, 3.0, 0.9, compute.max()])
    rng = np.random.default_rng(seed)
    best_r2, best_params = -np.inf, None
    for _ in range(n_restarts):
        p0 = [
            rng.uniform(0.1, 5.0),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.0, max(error.min() - 1e-3, 0.0)),
            rng.uniform(0.0, compute.max()),
        ]
        try:
            popt, _ = curve_fit(
                _double_saturating_power_law, compute, error,
                p0=p0, bounds=bounds, maxfev=20000,
            )
        except RuntimeError:
            continue
        pred = _double_saturating_power_law(compute, *popt)
        ss_res = np.sum((error - pred) ** 2)
        ss_tot = np.sum((error - np.mean(error)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_params = r2, popt
    if best_params is None:
        raise RuntimeError("Scaling-law fit failed to converge from any random restart.")
    return best_params, best_r2


def plot_pareto_frontier(df: DataFrame) -> None:
    # Compute-vs-error Pareto plot, styled after Fig. 2 of "Scaling Vision
    # Transformers" (Zhai et al. 2022): color = model size (coolwarm, blue
    # small -> red large), marker shape = training dataset, marker size =
    # dataset fraction, lines connect same model/dataset/bg_range runs.
    set_style()

    data = df.copy()
    data["error_rate"] = 1.0 - data["max_val_acc1"].astype(float)
    data["compute_eflops"] = data["total_flops"].astype(float) / 1e18
    data["bg_key"] = data["bg_range"].fillna("none")

    models = [m for m in _MODEL_SIZE_ORDER if m in data["model_name"].unique()]
    if not models:
        raise ValueError("No known models (from _MODEL_SIZE_ORDER) found in database.")
    cmap = plt.cm.coolwarm
    model_colors = {m: cmap(t) for m, t in zip(models, _model_color_positions(len(models)))}

    datasets = {
        "fornet/all/1.0": ("ImageNet", "o"),
        "fornet/all/cos": ("ForNet", "s"),
    }
    frontier_style = {
        "fornet/all/1.0": ("ImageNet", "black", "--"),
        "fornet/all/cos": ("ForNet", "#555555", "-."),
    }

    frac_min = data["train_dataset_fraction"].min()
    frac_max = data["train_dataset_fraction"].max()

    def marker_area(fraction):
        return 35.0 + 260.0 * (fraction - frac_min) / (frac_max - frac_min)

    fig, ax = plt.subplots(figsize=(12, 9))

    for model in models:
        color = model_colors[model]
        for dataset_name, (_, marker) in datasets.items():
            sub = data[(data["model_name"] == model) & (data["train_dataset_name"] == dataset_name)]
            if sub.empty:
                continue
            for _, bg_sub in sub.groupby("bg_key"):
                bg_sub = bg_sub.sort_values("train_dataset_fraction")
                x = bg_sub["compute_eflops"].to_numpy(float)
                y = bg_sub["error_rate"].to_numpy(float)
                sizes = marker_area(bg_sub["train_dataset_fraction"].to_numpy(float))
                ax.plot(x, y, "-", color=color, alpha=0.3, lw=1.0, zorder=2)
                ax.scatter(
                    x, y, s=sizes, marker=marker, color=color,
                    edgecolor="#333333", linewidth=0.6, alpha=0.92, zorder=3,
                )

    fit_summaries = []
    for dataset_name, (label, line_color, ls) in frontier_style.items():
        sub = data[data["train_dataset_name"] == dataset_name]
        fx, fy = _pareto_frontier(sub["compute_eflops"].to_numpy(float), sub["error_rate"].to_numpy(float))
        (a, b, c, d), r2 = _fit_scaling_law(fx, fy)
        x_smooth = np.geomspace(fx.min(), fx.max(), 200)
        y_smooth = _double_saturating_power_law(x_smooth, a, b, c, d)
        ax.plot(x_smooth, y_smooth, ls, color=line_color, lw=2.3, zorder=4)
        fit_summaries.append(
            rf"{label}: $E={c:.3f}+{a:.3f}\,(C+{d:.2f})^{{-{b:.2f}}}$  ($R^2={r2:.3f}$, $n={len(fx)}$ frontier pts)"
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training compute $C$ (EFLOPs, i.e. $10^{18}$ FLOPs)")
    ax.set_ylabel("Top-1 error rate $E = 1 - $ Val/Acc1")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.15)
    ax.set_title(
        "Compute-Performance Pareto Frontier: ImageNet vs. ForNet\n"
        "(cf. Fig. 2, “Scaling Vision Transformers”, Zhai et al. 2022)",
        fontsize=14, fontweight="bold",
    )

    ax.text(
        0.02, 0.03,
        "\n".join(fit_summaries),
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999", alpha=0.92),
    )

    model_handles = [
        Line2D([], [], marker="o", color=model_colors[m], ls="", markersize=9, markeredgecolor="#333333")
        for m in models
    ]
    legend_models = ax.legend(
        model_handles, models, title="Model", loc="upper right",
        fontsize=8, title_fontsize=9, framealpha=0.95,
    )
    ax.add_artist(legend_models)

    dataset_handles = [Line2D([], [], marker=marker, color="black", ls="", markersize=8) for _, marker in datasets.values()]
    dataset_labels = [label for label, _ in datasets.values()]
    fit_handles = [Line2D([], [], color=color, ls=ls, lw=2.3) for _, color, ls in frontier_style.values()]
    fit_labels = [f"{label} frontier fit" for label, _, _ in frontier_style.values()]
    legend_shapes = ax.legend(
        dataset_handles + fit_handles, dataset_labels + fit_labels,
        title="Training data", loc="upper right", bbox_to_anchor=(0.99, 0.72),
        fontsize=8, title_fontsize=9, framealpha=0.95,
    )
    ax.add_artist(legend_shapes)

    frac_ref = sorted(data["train_dataset_fraction"].unique())
    size_handles = [
        Line2D([], [], marker="o", color="gray", ls="", markersize=np.sqrt(marker_area(f)))
        for f in frac_ref
    ]
    size_labels = [f"{f:.2f}" for f in frac_ref]
    ax.legend(
        size_handles, size_labels, title="Dataset fraction", loc="upper left",
        fontsize=8, title_fontsize=9, framealpha=0.95, labelspacing=1.1,
    )

    fig.tight_layout()
    fig.savefig(PDF_OUTPUTS_DIR / "pareto_frontier.pdf")
    fig.savefig(IMG_OUTPUT_DIR / "pareto_frontier.png")
    plt.show()
