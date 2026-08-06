import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import seaborn as sns
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


def _log_double_saturating_power_law(compute, a, b, c, d):
    return np.log(_double_saturating_power_law(compute, a, b, c, d))


def _fit_scaling_law(compute, error, n_restarts=25, seed=0):
    # Random multi-start in log-space: minimizes relative squared log-error
    # so that high compute (low error) points are weighted equally with
    # low compute (high error) points on a logarithmic scale.
    bounds = ([1e-6, 0.01, 0.0, 0.0], [100.0, 3.0, 0.9, compute.max()])
    rng = np.random.default_rng(seed)
    best_r2, best_params = -np.inf, None
    log_error = np.log(error)
    for _ in range(n_restarts):
        p0 = [
            rng.uniform(0.1, 5.0),
            rng.uniform(0.05, 1.5),
            rng.uniform(0.0, max(error.min() - 1e-3, 0.0)),
            rng.uniform(0.0, compute.max()),
        ]
        try:
            popt, _ = curve_fit(
                _log_double_saturating_power_law, compute, log_error,
                p0=p0, bounds=bounds, maxfev=20000,
            )
        except RuntimeError:
            continue
        log_pred = _log_double_saturating_power_law(compute, *popt)
        ss_res = np.sum((log_error - log_pred) ** 2)
        ss_tot = np.sum((log_error - np.mean(log_error)) ** 2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2, best_params = r2, popt
    if best_params is None:
        raise RuntimeError("Scaling-law fit failed to converge from any random restart.")
    return best_params, best_r2


def _configure_pareto_ax(ax):
    """Apply shared axis styling for a Pareto subplot."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.grid(True, which="major", ls="--", alpha=0.4)
    ax.grid(True, which="minor", ls=":", alpha=0.15)


def plot_pareto_frontier(df: DataFrame) -> None:
    """
    Compute-vs-error Pareto plot split into two side-by-side subplots:
      - Left:  ImageNet (fornet/all/1.0)
      - Right: ForNet   (fornet/all/cos)

    Marker colour = model size (coolwarm, blue small -> red large).
    ImageNet subplot: marker shape = circle, marker size = dataset fraction.
    ForNet subplot:   marker shape encodes bg_range; thin solid lines connect
                      same-bg_range points across fg_ranges; dotted lines
                      connect same-fg_range points across bg_ranges (iso-
                      compute brackets showing bg_range sensitivity).
                      The Pareto frontier is computed over ALL ForNet points
                      (best error per compute budget across all bg_ranges).
    """
    set_style()

    data = df.copy()
    data["error_rate"]     = 1.0 - data["max_val_acc1"].astype(float)
    data["compute_eflops"] = data["total_flops"].astype(float) / 1e18
    data["bg_key"]         = data["bg_range"].fillna("none")

    # ── Model colours (coolwarm) ────────────────────────────────────────────
    models = [m for m in _MODEL_SIZE_ORDER if m in data["model_name"].unique()]
    if not models:
        raise ValueError("No known models (from _MODEL_SIZE_ORDER) found in database.")
    cmap = plt.cm.coolwarm
    model_colors = {m: cmap(t) for m, t in zip(models, _model_color_positions(len(models)))}

    # ── Split datasets ──────────────────────────────────────────────────────
    imgnet_data = data[data["train_dataset_name"] == "fornet/all/1.0"].copy()
    fornet_data = data[data["train_dataset_name"] == "fornet/all/cos"].copy()

    # ── ForNet: bg_range palette and marker shapes ──────────────────────────
    # Natural sort: 0-10, 0-25, 0-50, 0-100
    bg_ranges_sorted = sorted(
        [bg for bg in fornet_data["bg_key"].unique() if bg != "none"],
        key=lambda x: int(x.split("-")[1]),
    )
    # Colorblind-friendly categorical palette for bg_range
    bg_palette = sns.color_palette("colorblind", n_colors=len(bg_ranges_sorted))
    bg_colors  = {bg: bg_palette[i] for i, bg in enumerate(bg_ranges_sorted)}  # noqa: F841
    # Distinct marker shapes for a second visual channel on bg_range
    _bg_marker_pool = ["o", "s", "^", "D"]
    bg_markers = {bg: _bg_marker_pool[i % len(_bg_marker_pool)]
                  for i, bg in enumerate(bg_ranges_sorted)}

    # ── Dataset fraction -> marker area ────────────────────────────────────
    frac_min = data["train_dataset_fraction"].min()
    frac_max = data["train_dataset_fraction"].max()

    def marker_area(fraction):
        return 35.0 + 260.0 * (fraction - frac_min) / (frac_max - frac_min)

    # Figure: two side-by-side subplots, shared y-axis, constrained layout
    fig, (ax_img, ax_fn) = plt.subplots(
        1, 2,
        figsize=(18, 8),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.06},
    )

    # ════════════════════════════════════════════════════════════════════════
    # LEFT SUBPLOT — ImageNet
    # ════════════════════════════════════════════════════════════════════════
    for model in models:
        color = model_colors[model]
        sub = imgnet_data[imgnet_data["model_name"] == model].sort_values("train_dataset_fraction")
        if sub.empty:
            continue
        x     = sub["compute_eflops"].to_numpy(float)
        y     = sub["error_rate"].to_numpy(float)
        sizes = marker_area(sub["train_dataset_fraction"].to_numpy(float))
        ax_img.plot(x, y, "-", color=color, alpha=0.30, lw=1.2, zorder=2)
        ax_img.scatter(
            x, y, s=sizes, marker="o", color=color,
            edgecolor="#333333", linewidth=0.6, alpha=0.92, zorder=3,
        )


    # Pareto frontier fit — ImageNet
    fx_img, fy_img = _pareto_frontier(
        imgnet_data["compute_eflops"].to_numpy(float),
        imgnet_data["error_rate"].to_numpy(float),
    )
    (a, b, c, d), r2_img = _fit_scaling_law(fx_img, fy_img)
    x_sm = np.geomspace(fx_img.min(), fx_img.max(), 300)
    ax_img.plot(x_sm, _double_saturating_power_law(x_sm, a, b, c, d),
                "--", color="#1a1a2e", lw=2.5, zorder=4)

    _configure_pareto_ax(ax_img)
    ax_img.set_ylabel("Top-1 error rate  $E = 1 - $ Val/Acc1", fontsize=11)
    ax_img.set_title("ImageNet", fontsize=14, fontweight="bold", pad=10)
    ax_img.text(
        0.03, 0.04,
        rf"$E={c:.3f}+{a:.3f}\,(C+{d:.2f})^{{-{b:.2f}}}$" + "\n" +
        rf"$R^2={r2_img:.3f}$,  $n={len(fx_img)}$ frontier pts",
        transform=ax_img.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#aaaaaa", alpha=0.92),
    )

    # =====================================================================
    # RIGHT SUBPLOT: ForNet
    # For each (model, fg_range) there are multiple bg_range rows sharing
    # the same compute budget.
    #  1. Solid lines:  same bg_range across fg_ranges (scaling trend).
    #  2. Dotted lines: same fg_range across bg_ranges (iso-compute bracket).
    #  3. Scatter each point: colour = model, shape = bg_range, size = frac.
    #  4. Pareto frontier over ALL ForNet points (best error per budget).
    # =====================================================================
    for model in models:
        color = model_colors[model]
        model_sub = fornet_data[fornet_data["model_name"] == model]
        if model_sub.empty:
            continue

        # 1. Solid lines: same bg_range across fg_ranges
        for bg_key, bg_sub in model_sub.groupby("bg_key"):
            bg_sub = bg_sub.sort_values("train_dataset_fraction")
            ax_fn.plot(
                bg_sub["compute_eflops"].to_numpy(float),
                bg_sub["error_rate"].to_numpy(float),
                "-", color=color, alpha=0.25, lw=0.9, zorder=2,
            )

        # 2. Dotted lines: same fg_range across bg_ranges (iso-compute bracket)
        for fg_key, fg_sub in model_sub.groupby("fg_range"):
            fg_sub_sorted = fg_sub.sort_values(
                "bg_key",
                key=lambda s: s.map(lambda x: int(x.split("-")[1]) if x != "none" else 0),
            )
            ax_fn.plot(
                fg_sub_sorted["compute_eflops"].to_numpy(float),
                fg_sub_sorted["error_rate"].to_numpy(float),
                ":", color=color, alpha=0.20, lw=0.8, zorder=2,
            )

        # 3. Scatter: colour = model, shape = bg_range, size = dataset fraction
        for bg_key, bg_sub in model_sub.groupby("bg_key"):
            if bg_key not in bg_markers:
                continue
            x     = bg_sub["compute_eflops"].to_numpy(float)
            y     = bg_sub["error_rate"].to_numpy(float)
            sizes = marker_area(bg_sub["train_dataset_fraction"].to_numpy(float))
            ax_fn.scatter(
                x, y, s=sizes, marker=bg_markers[bg_key], color=color,
                edgecolor="#333333", linewidth=0.6, alpha=0.92, zorder=3,
            )

    # 4. Pareto frontier + fit for ForNet (global: all bg_ranges compete)
    fx_fn, fy_fn = _pareto_frontier(
        fornet_data["compute_eflops"].to_numpy(float),
        fornet_data["error_rate"].to_numpy(float),
    )
    (a2, b2, c2, d2), r2_fn = _fit_scaling_law(fx_fn, fy_fn)
    x_sm2 = np.geomspace(fx_fn.min(), fx_fn.max(), 300)
    ax_fn.plot(x_sm2, _double_saturating_power_law(x_sm2, a2, b2, c2, d2),
               "-.", color="#1a1a2e", lw=2.5, zorder=4)

    _configure_pareto_ax(ax_fn)
    ax_fn.set_title("ForNet", fontsize=14, fontweight="bold", pad=10)
    ax_fn.tick_params(axis="y", labelleft=False)   # shared y-axis; hide duplicate labels
    ax_fn.text(
        0.03, 0.04,
        rf"$E={c2:.3f}+{a2:.3f}\,(C+{d2:.2f})^{{-{b2:.2f}}}$" + "\n" +
        rf"$R^2={r2_fn:.3f}$,  $n={len(fx_fn)}$ frontier pts",
        transform=ax_fn.transAxes, ha="left", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#aaaaaa", alpha=0.92),
    )

    # Shared x-axis label centred between both subplots
    fig.text(
        0.5, 0.01,
        r"Training compute $C$ (EFLOPs, i.e. $10^{18}$ FLOPs)",
        ha="center", va="bottom", fontsize=11,
    )

    fig.suptitle(
        "Compute\u2013Performance Pareto Frontier: ImageNet vs. ForNet\n"
        "(cf. Fig. 2, \u201cScaling Vision Transformers\u201d, Zhai et al. 2022)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # =====================================================================
    # LEGENDS
    # =====================================================================

    # 1. Model-size colour legend (left subplot, upper right)
    model_handles = [
        Line2D([], [], marker="o", color=model_colors[m], ls="",
               markersize=9, markeredgecolor="#333333", label=m)
        for m in models
    ]
    leg_model = ax_img.legend(
        handles=model_handles, title="Model size",
        loc="upper right", fontsize=8, title_fontsize=9, framealpha=0.95,
    )
    ax_img.add_artist(leg_model)

    # 2. Dataset-fraction / marker-size legend (left subplot, mid-right)
    frac_ref = sorted(data["train_dataset_fraction"].unique())
    size_handles = [
        Line2D([], [], marker="o", color="gray", ls="",
               markersize=np.sqrt(marker_area(f)), label=f"{f:.2f}")
        for f in frac_ref
    ]
    leg_size = ax_img.legend(
        handles=size_handles, title="Dataset fraction",
        loc="upper right", bbox_to_anchor=(0.99, 0.60),
        fontsize=8, title_fontsize=9, framealpha=0.95, labelspacing=1.1,
    )
    ax_img.add_artist(leg_size)

    # 3. Pareto frontier fit line (left subplot, lower left)
    leg_fit_img = ax_img.legend(
        handles=[Line2D([], [], color="#1a1a2e", ls="--", lw=2.5,
                        label="Pareto frontier fit")],
        loc="lower left", fontsize=8, framealpha=0.95,
    )
    ax_img.add_artist(leg_fit_img)

    # 4. bg_range shape legend (right subplot, upper right)
    bg_handles = [
        Line2D([], [], marker=bg_markers[bg], color="#444444", ls="",
               markersize=8, markeredgecolor="#333333", label=f"bg {bg}")
        for bg in bg_ranges_sorted
    ]
    leg_bg = ax_fn.legend(
        handles=bg_handles, title="bg_range",
        loc="upper right", fontsize=8, title_fontsize=9, framealpha=0.95,
    )
    ax_fn.add_artist(leg_bg)

    # 5. Pareto frontier fit line (right subplot, lower left)
    ax_fn.legend(
        handles=[Line2D([], [], color="#1a1a2e", ls="-.", lw=2.5,
                        label="Pareto frontier fit")],
        loc="lower left", fontsize=8, framealpha=0.95,
    )

    # Save
    fig.savefig(PDF_OUTPUTS_DIR / "pareto_frontier.pdf", bbox_inches="tight")
    fig.savefig(IMG_OUTPUT_DIR  / "pareto_frontier.png", bbox_inches="tight")
    plt.show()
