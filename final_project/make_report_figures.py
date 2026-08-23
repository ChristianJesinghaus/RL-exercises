"""Create publication-ready vector figures for the final report."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "DQN": "#4C78A8",
    "RND": "#F58518",
    "LP-RND": "#2A9D8F",
}
MARKERS = {"DQN": "o", "RND": "s", "LP-RND": "^"}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "lines.linewidth": 2.0,
        "figure.dpi": 180,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def short_method(method: str) -> str:
    return {"dqn": "DQN", "rnd": "RND", "lp_rnd": "LP-RND"}[method]


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def bootstrap_curve(values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap seeds; values has shape seeds x checkpoints."""
    n = values.shape[0]
    draws = values[rng.integers(0, n, size=(2000, n)), :].mean(axis=1)
    return values.mean(axis=0), np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def metric_frame(environment: str, filename: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / filename)
    frame["environment"] = environment
    frame["short_method"] = frame["method"].map(short_method)
    return frame


def plot_results_overview() -> None:
    noisy_eval = pd.read_csv(DATA / "main_evaluations_all.csv")
    noisy_eval = noisy_eval[
        ((noisy_eval["method"] == "dqn") & (noisy_eval["beta"] == 0.0))
        | ((noisy_eval["method"].isin(["rnd", "lp_rnd"])) & (noisy_eval["beta"] == 0.01))
    ].copy()
    noisy_eval["short_method"] = noisy_eval["method"].map(short_method)

    noisy = metric_frame("Noisy TV", "main_aggregate_summary.csv")
    noisy = noisy[
        ((noisy["method"] == "dqn") & (noisy["beta"] == 0.0))
        | ((noisy["method"].isin(["rnd", "lp_rnd"])) & (noisy["beta"] == 0.01))
    ]
    clean = metric_frame("Clean", "clean_aggregate_summary.csv")
    aggregate = pd.concat([clean, noisy], ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.55))
    ax_curve, ax_auc, ax_signal, ax_visit = axes.flat
    rng = np.random.default_rng(20260729)

    for method in ["DQN", "RND", "LP-RND"]:
        subset = noisy_eval[noisy_eval["short_method"] == method]
        pivot = subset.pivot(index="seed", columns="step", values="success_rate")
        steps = pivot.columns.to_numpy()
        mean, low, high = bootstrap_curve(pivot.to_numpy(), rng)
        ax_curve.plot(steps / 1000, mean, color=COLORS[method], label=method)
        ax_curve.fill_between(steps / 1000, low, high, color=COLORS[method], alpha=0.16, linewidth=0)
    ax_curve.set_title("a  Noisy-TV task performance")
    ax_curve.set_xlabel("Environment steps (thousands)")
    ax_curve.set_ylabel("Greedy success rate")
    ax_curve.set_xlim(0, 100)
    ax_curve.set_ylim(-0.03, 1.03)
    ax_curve.set_yticks(np.linspace(0, 1, 6))
    ax_curve.legend(loc="upper left", ncol=3, frameon=False, columnspacing=0.9, handlelength=1.7)

    offsets = {"Clean": -0.13, "Noisy TV": 0.13}
    env_markers = {"Clean": "o", "Noisy TV": "D"}
    xpos = {"DQN": 0, "RND": 1, "LP-RND": 2}
    for env in ["Clean", "Noisy TV"]:
        for method in ["DQN", "RND", "LP-RND"]:
            row = aggregate[(aggregate.environment == env) & (aggregate.short_method == method)].iloc[0]
            x = xpos[method] + offsets[env]
            y = row.success_rate_auc_mean
            err = np.array([[y - row.success_rate_auc_ci_low], [row.success_rate_auc_ci_high - y]])
            ax_auc.errorbar(
                x, y, yerr=err, fmt=env_markers[env], color=COLORS[method],
                markeredgecolor="white", markeredgewidth=0.6, markersize=6,
                capsize=2.8, linewidth=1.4,
            )
    ax_auc.set_title("b  Success-rate AUC")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_xticks([0, 1, 2], ["DQN", "RND", "LP-RND"])
    ax_auc.set_ylim(-0.03, 0.72)
    ax_auc.legend(
        handles=[
            Line2D([], [], marker="o", color="#555555", linestyle="none", label="Clean"),
            Line2D([], [], marker="D", color="#555555", linestyle="none", label="Noisy TV"),
        ],
        loc="upper right", frameon=False, handletextpad=0.4,
    )

    intrinsic = aggregate[aggregate.short_method.isin(["RND", "LP-RND"])]
    xpos_signal = {"RND": 0, "LP-RND": 1}
    for env in ["Clean", "Noisy TV"]:
        for method in ["RND", "LP-RND"]:
            row = intrinsic[(intrinsic.environment == env) & (intrinsic.short_method == method)].iloc[0]
            x = xpos_signal[method] + offsets[env]
            y = row.tail_inside_outside_ratio_mean
            err = np.array([[y - row.tail_inside_outside_ratio_ci_low], [row.tail_inside_outside_ratio_ci_high - y]])
            ax_signal.errorbar(
                x, y, yerr=err, fmt=env_markers[env], color=COLORS[method],
                markeredgecolor="white", markeredgewidth=0.6, markersize=6,
                capsize=2.8, linewidth=1.4,
            )
    ax_signal.axhline(1.0, color="#666666", linewidth=1, linestyle="--", zorder=0)
    ax_signal.set_yscale("log")
    ax_signal.set_title("c  Tail intrinsic selectivity")
    ax_signal.set_ylabel("Inside / outside TV signal")
    ax_signal.set_xticks([0, 1], ["RND", "LP-RND"])
    ax_signal.set_ylim(0.35, 100)
    ax_signal.text(0.98, 1.08, "equal signal", color="#666666", fontsize=7.5, ha="right", va="bottom")

    for env in ["Clean", "Noisy TV"]:
        for method in ["DQN", "RND", "LP-RND"]:
            row = aggregate[(aggregate.environment == env) & (aggregate.short_method == method)].iloc[0]
            x = xpos[method] + offsets[env]
            y = row.training_tv_fraction_mean
            err = np.array([[y - row.training_tv_fraction_ci_low], [row.training_tv_fraction_ci_high - y]])
            ax_visit.errorbar(
                x, y, yerr=err, fmt=env_markers[env], color=COLORS[method],
                markeredgecolor="white", markeredgewidth=0.6, markersize=6,
                capsize=2.8, linewidth=1.4,
            )
    ax_visit.set_title("d  Distractor-region visitation")
    ax_visit.set_ylabel("Training fraction in TV region")
    ax_visit.set_xticks([0, 1, 2], ["DQN", "RND", "LP-RND"])
    ax_visit.set_ylim(0.04, 0.145)

    for ax in axes.flat:
        ax.tick_params(direction="out", length=3)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.96, bottom=0.10, wspace=0.30, hspace=0.43)
    save_both(fig, "results_overview")


def plot_beta_sensitivity() -> None:
    frame = pd.read_csv(DATA / "main_aggregate_summary.csv")
    frame = frame[frame.method.isin(["rnd", "lp_rnd"])].copy()
    frame["short_method"] = frame.method.map(short_method)
    fig, ax = plt.subplots(figsize=(5.2, 2.35))
    for method in ["RND", "LP-RND"]:
        d = frame[frame.short_method == method].sort_values("beta")
        y = d.success_rate_auc_mean.to_numpy()
        lo = d.success_rate_auc_ci_low.to_numpy()
        hi = d.success_rate_auc_ci_high.to_numpy()
        ax.errorbar(
            d.beta, y, yerr=np.vstack([y - lo, hi - y]),
            color=COLORS[method], marker=MARKERS[method], capsize=3,
            markersize=6, label=method,
        )
    ax.axhline(0.438, color=COLORS["DQN"], linestyle="--", linewidth=1.4, label="DQN (0.438)")
    ax.set_xlabel(r"Intrinsic-reward scale $\beta$")
    ax.set_ylabel("Success-rate AUC")
    ax.set_xticks([0.01, 0.05, 0.10])
    ax.set_ylim(-0.03, 0.68)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(pad=0.5)
    save_both(fig, "beta_sensitivity")


def plot_environment() -> None:
    fig, ax = plt.subplots(figsize=(4.85, 3.25))
    size = 11
    ax.add_patch(Rectangle((0, 0), size, size, facecolor="#FAFAF7", edgecolor="#333333", linewidth=1.2))

    # Outer border and the two FourRooms dividing walls. Coordinates are cells.
    wall_cells = set()
    for x in range(size):
        wall_cells.add((x, 0))
        wall_cells.add((x, size - 1))
    for y in range(size):
        wall_cells.add((0, y))
        wall_cells.add((size - 1, y))
    doors = {(5, 2), (5, 8), (2, 5), (9, 5)}
    for y in range(1, size - 1):
        if (5, y) not in doors:
            wall_cells.add((5, y))
    for x in range(1, size - 1):
        if (x, 5) not in doors:
            wall_cells.add((x, 5))

    for x, y in wall_cells:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#3D405B", edgecolor="white", linewidth=0.25))

    tv_cells = {(2, 2), (2, 3), (3, 2), (3, 3)}
    for x, y in tv_cells:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#F4A261", edgecolor="white", linewidth=0.6, alpha=0.86))

    ax.scatter([2.5], [2.5], s=72, marker=">", color=COLORS["DQN"], edgecolor="white", linewidth=0.8, zorder=5)
    ax.scatter([9.5], [8.5], s=100, marker="*", color=COLORS["LP-RND"], edgecolor="white", linewidth=0.7, zorder=5)
    ax.annotate("start", (2.5, 2.5), xytext=(1.15, 1.2), arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8), fontsize=8.5)
    ax.annotate("goal", (9.5, 8.5), xytext=(8.1, 9.35), arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8), fontsize=8.5)
    ax.text(2.98, 3.55, "noisy TV", ha="center", va="bottom", fontsize=8.5, color="#8A4F08")
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.1)
    save_both(fig, "environment_schematic")


if __name__ == "__main__":
    plot_results_overview()
    plot_beta_sensitivity()
    plot_environment()
