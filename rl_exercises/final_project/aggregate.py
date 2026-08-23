"""Aggregate completed runs into CSV summaries, curves, and heatmaps.
ChatGPT 5.5 Thinking was used for parts of code production, review and overall help in this file.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_exercises.final_project.envs import FixedFourRoomsEnv


@dataclass
class RunData:
    directory: Path
    config: dict[str, Any]
    summary: dict[str, Any]
    evaluations: pd.DataFrame
    diagnostics: pd.DataFrame
    visitation: np.ndarray

    @property
    def label(self) -> str:
        return method_label(self.config["method"], float(self.config["beta"]))


def method_label(method: str, beta: float) -> str:
    if method == "dqn":
        return "DQN"
    name = "RND" if method == "rnd" else "LP-RND"
    return f"{name} (beta={beta:g})"


def label_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def discover_runs(root: str | Path) -> list[RunData]:
    root_path = Path(root).expanduser().resolve()
    run_root = root_path / "runs" if (root_path / "runs").is_dir() else root_path
    runs: list[RunData] = []
    for summary_path in sorted(run_root.glob("*/summary.json")):
        directory = summary_path.parent
        required = (
            directory / "config.json",
            directory / "evaluations.csv",
            directory / "diagnostics.csv",
            directory / "visitation.npy",
        )
        if not all(path.exists() for path in required):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            continue
        config = json.loads(required[0].read_text(encoding="utf-8"))
        runs.append(
            RunData(
                directory=directory,
                config=config,
                summary=summary,
                evaluations=pd.read_csv(required[1]),
                diagnostics=pd.read_csv(required[2]),
                visitation=np.load(required[3]),
            )
        )
    if not runs:
        raise FileNotFoundError(f"No complete runs found below {run_root}")
    return runs


def _bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int = 2_000,
) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    if finite.size == 1:
        return mean, mean, mean
    indices = rng.integers(0, finite.size, size=(repetitions, finite.size))
    bootstrap_means = finite[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))
    return mean, float(low), float(high)


def _combine_frames(runs: list[RunData], attribute: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        frame = getattr(run, attribute).copy()
        frame["method"] = run.config["method"]
        frame["beta"] = 0.0 if run.config["method"] == "dqn" else run.config["beta"]
        frame["seed"] = run.config["seed"]
        frame["label"] = run.label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _curve_statistics(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    rng = np.random.default_rng(20260729)
    records: list[dict[str, Any]] = []
    for (label, step), group in frame.groupby(["label", "step"], sort=True):
        mean, low, high = _bootstrap_interval(
            group[value_column].to_numpy(dtype=float), rng
        )
        records.append(
            {
                "label": label,
                "step": step,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n": group["seed"].nunique(),
            }
        )
    return pd.DataFrame(records)


def _plot_curve(
    frame: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    stats = _curve_statistics(frame, value_column)
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for label, group in stats.groupby("label", sort=True):
        group = group.sort_values("step")
        axis.plot(group["step"], group["mean"], label=label, linewidth=2)
        axis.fill_between(
            group["step"],
            group["ci_low"],
            group["ci_high"],
            alpha=0.18,
        )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel(ylabel)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_intrinsic_inside_outside(
    diagnostics: pd.DataFrame, output_path: Path
) -> None:
    intrinsic = diagnostics[diagnostics["method"] != "dqn"].copy()
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color_index, (label, group) in enumerate(intrinsic.groupby("label", sort=True)):
        color = colors[color_index % len(colors)]
        for zone, style in (("inside", "-"), ("outside", "--")):
            column = f"intrinsic_normalized_{zone}"
            stats = _curve_statistics(group, column)
            stats = stats.sort_values("step")
            axis.plot(
                stats["step"],
                stats["mean"],
                linestyle=style,
                color=color,
                linewidth=2,
                label=f"{label}, {zone} TV",
            )
            axis.fill_between(
                stats["step"],
                stats["ci_low"],
                stats["ci_high"],
                color=color,
                alpha=0.10,
            )
    axis.set_xlabel("Environment steps")
    axis.set_ylabel("Normalized intrinsic signal")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _wall_mask() -> np.ndarray:
    size = FixedFourRoomsEnv.size
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[1:-1, FixedFourRoomsEnv.wall_coordinate] = True
    mask[FixedFourRoomsEnv.wall_coordinate, 1:-1] = True
    for x, y in FixedFourRoomsEnv.openings:
        mask[y, x] = False
    return mask


def _plot_heatmap(visitation: np.ndarray, label: str, output_path: Path) -> None:
    wall_mask = _wall_mask()
    values = np.log1p(visitation.astype(float))
    values[wall_mask] = np.nan
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("black")

    figure, axis = plt.subplots(figsize=(5.4, 5.0))
    image = axis.imshow(values, cmap=color_map, origin="upper")
    for x, y in FixedFourRoomsEnv.tv_positions:
        axis.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=1.5,
            )
        )
    start_x, start_y = FixedFourRoomsEnv.start_pos
    goal_x, goal_y = FixedFourRoomsEnv.goal_pos
    axis.text(
        start_x,
        start_y,
        "S",
        ha="center",
        va="center",
        color="white",
        weight="bold",
    )
    axis.text(
        goal_x,
        goal_y,
        "G",
        ha="center",
        va="center",
        color="white",
        weight="bold",
    )
    axis.set_title(f"{label}: log(1 + visits)")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_xticks(range(FixedFourRoomsEnv.size))
    axis.set_yticks(range(FixedFourRoomsEnv.size))
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _run_summary_table(runs: list[RunData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        diagnostics = run.diagnostics
        tail_start = diagnostics["step"].max() * 0.8
        tail = diagnostics[diagnostics["step"] >= tail_start]
        inside = float(tail["intrinsic_normalized_inside"].mean())
        outside = float(tail["intrinsic_normalized_outside"].mean())
        ratio = (
            inside / outside
            if np.isfinite(inside) and np.isfinite(outside) and outside > 0.0
            else float("nan")
        )
        rows.append(
            {
                "label": run.label,
                "method": run.config["method"],
                "beta": (
                    0.0 if run.config["method"] == "dqn" else float(run.config["beta"])
                ),
                "seed": int(run.config["seed"]),
                "final_eval_success_rate": run.summary["final_eval_success_rate"],
                "best_eval_success_rate": run.summary["best_eval_success_rate"],
                "success_rate_auc": run.summary["success_rate_auc"],
                "final_eval_extrinsic_return": run.summary[
                    "final_eval_extrinsic_return"
                ],
                "first_reward_step": run.summary["first_reward_step"],
                "third_reward_step": run.summary["third_reward_step"],
                "training_tv_fraction": run.summary["training_tv_fraction"],
                "position_coverage": run.summary["position_coverage"],
                "tail_intrinsic_inside": inside,
                "tail_intrinsic_outside": outside,
                "tail_inside_outside_ratio": ratio,
                "elapsed_seconds": run.summary["elapsed_seconds"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_eval_success_rate",
        "best_eval_success_rate",
        "success_rate_auc",
        "final_eval_extrinsic_return",
        "first_reward_step",
        "third_reward_step",
        "training_tv_fraction",
        "position_coverage",
        "tail_intrinsic_inside",
        "tail_intrinsic_outside",
        "tail_inside_outside_ratio",
        "elapsed_seconds",
    ]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260729)
    for (label, method, beta), group in per_run.groupby(
        ["label", "method", "beta"], sort=True
    ):
        row: dict[str, Any] = {
            "label": label,
            "method": method,
            "beta": beta,
            "n_seeds": group["seed"].nunique(),
        }
        for metric in metrics:
            mean, low, high = _bootstrap_interval(
                group[metric].to_numpy(dtype=float), rng
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_results(root: str | Path) -> Path:
    root_path = Path(root).expanduser().resolve()
    runs = discover_runs(root_path)
    output_dir = root_path / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    evaluations = _combine_frames(runs, "evaluations")
    diagnostics = _combine_frames(runs, "diagnostics")
    per_run = _run_summary_table(runs)
    aggregate = _aggregate_summary(per_run)

    evaluations.to_csv(output_dir / "evaluations_all.csv", index=False)
    diagnostics.to_csv(output_dir / "diagnostics_all.csv", index=False)
    per_run.to_csv(output_dir / "per_run_summary.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_summary.csv", index=False)

    _plot_curve(
        evaluations,
        "success_rate",
        "Greedy evaluation success rate",
        plots_dir / "success_rate.png",
        ylim=(-0.02, 1.02),
    )
    _plot_curve(
        evaluations,
        "mean_extrinsic_return",
        "Greedy evaluation extrinsic return",
        plots_dir / "extrinsic_return.png",
    )
    _plot_curve(
        diagnostics,
        "tv_fraction_cumulative",
        "Cumulative training fraction in TV region",
        plots_dir / "tv_fraction.png",
        ylim=(-0.02, 1.02),
    )
    _plot_curve(
        diagnostics,
        "position_coverage",
        "Position coverage (analysis only)",
        plots_dir / "position_coverage.png",
        ylim=(-0.02, 1.02),
    )
    _plot_intrinsic_inside_outside(
        diagnostics, plots_dir / "intrinsic_inside_outside.png"
    )

    for label in sorted({run.label for run in runs}):
        matching = [run.visitation for run in runs if run.label == label]
        _plot_heatmap(
            np.sum(matching, axis=0),
            label,
            plots_dir / f"heatmap_{label_slug(label)}.png",
        )

    manifest = {
        "complete_runs": len(runs),
        "labels": sorted({run.label for run in runs}),
        "files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Aggregated {len(runs)} runs into {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        help="Sweep root containing runs/, or a directory of run folders.",
    )
    args = parser.parse_args()
    aggregate_results(args.results)


if __name__ == "__main__":
    main()
