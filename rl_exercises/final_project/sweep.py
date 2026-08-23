"""Run the smoke, pilot, main, or clean experiment matrix.

Preset sizes
------------
smoke: 3 short runs (one per method)
pilot: 9 runs (three methods x three seeds, beta=0.01)
main: 35 runs (DQN x 5 seeds; RND and LP-RND x 3 betas x 5 seeds)
clean: 15 runs (three methods x five seeds, beta=0.01, Noisy TV disabled)
ChatGPT 5.5 Thinking was used for parts of code production, review and overall help in this file.

"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from rl_exercises.final_project.experiment import ExperimentConfig

Preset = Literal["smoke", "pilot", "main", "clean"]

def beta_slug(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def run_name(method: str, beta: float, seed: int) -> str:
    if method == "dqn":
        return f"dqn_seed{seed:03d}"
    return f"{method}_beta{beta_slug(beta)}_seed{seed:03d}"


def preset_configs(preset: Preset, output_root: str | Path) -> list[ExperimentConfig]:
    """Build the exact run matrix for a named preset."""

    root = Path(output_root).resolve()
    configs: list[ExperimentConfig] = []

    if preset == "smoke":
        base = ExperimentConfig(
            total_steps=1_200,
            beta=0.01,
            learning_starts=64,
            batch_size=32,
            buffer_capacity=2_000,
            target_update_interval=200,
            epsilon_decay_steps=1_000,
            lp_snapshot_interval=200,
            eval_interval=600,
            eval_episodes=2,
            log_interval=300,
        )
        for method in ("dqn", "rnd", "lp_rnd"):
            beta = 0.0 if method == "dqn" else 0.01
            name = run_name(method, beta, 0)
            configs.append(
                replace(
                    base,
                    method=method,
                    beta=beta,
                    seed=0,
                    output_dir=str(root / "runs" / name),
                )
            )
        return configs

    if preset == "pilot":
        base = ExperimentConfig(
            total_steps=50_000,
            beta=0.01,
            eval_interval=5_000,
            eval_episodes=20,
            log_interval=1_000,
        )
        for seed in range(3):
            for method in ("dqn", "rnd", "lp_rnd"):
                beta = 0.0 if method == "dqn" else 0.01
                name = run_name(method, beta, seed)
                configs.append(
                    replace(
                        base,
                        method=method,
                        beta=beta,
                        seed=seed,
                        output_dir=str(root / "runs" / name),
                    )
                )
        return configs

    if preset == "clean":
        base = ExperimentConfig(
            total_steps=100_000,
            beta=0.01,
            noisy_tv=False,
            eval_interval=10_000,
            eval_episodes=20,
            log_interval=1_000,
        )
        for seed in range(5):
            for method in ("dqn", "rnd", "lp_rnd"):
                beta = 0.0 if method == "dqn" else 0.01
                name = run_name(method, beta, seed)
                configs.append(
                    replace(
                        base,
                        method=method,
                        beta=beta,
                        seed=seed,
                        output_dir=str(root / "runs" / name),
                    )
                )
        return configs    

    
    if preset == "main":
        base = ExperimentConfig(
            total_steps=100_000,
            eval_interval=10_000,
            eval_episodes=20,
            log_interval=1_000,
        )
        for seed in range(5):
            name = run_name("dqn", 0.0, seed)
            configs.append(
                replace(
                    base,
                    method="dqn",
                    beta=0.0,
                    seed=seed,
                    output_dir=str(root / "runs" / name),
                )
            )
        for method in ("rnd", "lp_rnd"):
            for beta in (0.01, 0.05, 0.1):
                for seed in range(5):
                    name = run_name(method, beta, seed)
                    configs.append(
                        replace(
                            base,
                            method=method,
                            beta=beta,
                            seed=seed,
                            output_dir=str(root / "runs" / name),
                        )
                    )
        return configs

    raise ValueError(f"Unknown preset: {preset}")


def _write_plan(root: Path, configs: list[ExperimentConfig]) -> list[Path]:
    config_dir = root / "run_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_paths: list[Path] = []
    plan_path = root / "run_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("run", "method", "beta", "seed", "total_steps", "output_dir"),
        )
        writer.writeheader()
        for config in configs:
            name = Path(config.output_dir).name
            config_path = config_dir / f"{name}.json"
            config_path.write_text(
                json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            config_paths.append(config_path)
            writer.writerow(
                {
                    "run": name,
                    "method": config.method,
                    "beta": config.beta,
                    "seed": config.seed,
                    "total_steps": config.total_steps,
                    "output_dir": config.output_dir,
                }
            )
    return config_paths


def _execute_config(config_path: Path) -> tuple[Path, int]:
    command = [
        sys.executable,
        "-m",
        "rl_exercises.final_project.experiment",
        "--config-json",
        str(config_path),
    ]
    completed = subprocess.run(command, check=False)
    return config_path, completed.returncode


def execute_sweep(
    preset: Preset,
    output_root: str | Path,
    *,
    jobs: int = 1,
    keep_going: bool = False,
    aggregate: bool = True,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"Sweep output directory is not empty: {root}. "
            "Use a fresh directory so existing runs are never overwritten."
        )
    root.mkdir(parents=True, exist_ok=True)
    configs = preset_configs(preset, root)
    config_paths = _write_plan(root, configs)

    failures: list[str] = []
    if jobs == 1:
        for index, config_path in enumerate(config_paths, start=1):
            print(
                f"\n=== {preset}: run {index}/{len(config_paths)} "
                f"({config_path.stem}) ===",
                flush=True,
            )
            _, return_code = _execute_config(config_path)
            if return_code != 0:
                failures.append(config_path.stem)
                if not keep_going:
                    break
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(_execute_config, path): path for path in config_paths
            }
            for future in as_completed(futures):
                config_path, return_code = future.result()
                print(
                    f"Finished {config_path.stem} with exit code {return_code}",
                    flush=True,
                )
                if return_code != 0:
                    failures.append(config_path.stem)

    status = {
        "preset": preset,
        "planned_runs": len(config_paths),
        "failed_runs": failures,
        "status": "failed" if failures else "complete",
    }
    (root / "sweep_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(
            "Sweep failed for: "
            + ", ".join(failures)
            + f". Partial results remain in {root}"
        )

    if aggregate:
        from rl_exercises.final_project.aggregate import aggregate_results

        aggregate_results(root)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=("smoke", "pilot", "main", "clean"))
    parser.add_argument(
        "--output-dir",
        help=("Fresh result directory. Default: results/<preset>_<YYYYmmdd_HHMMSS>"),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="Concurrent CPU runs. Keep the default 1 unless memory is plentiful.",
    )
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matrix without creating files or training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"results/{args.preset}_{timestamp}"
    configs = preset_configs(args.preset, output_dir)
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "run": Path(config.output_dir).name,
                        "method": config.method,
                        "beta": config.beta,
                        "seed": config.seed,
                        "total_steps": config.total_steps,
                        "noisy_tv": config.noisy_tv,
                    }
                    for config in configs
                ],
                indent=2,
            )
        )
        return

    root = execute_sweep(
        args.preset,
        output_dir,
        jobs=args.jobs,
        keep_going=args.keep_going,
        aggregate=not args.no_aggregate,
    )
    print(f"\nSweep complete: {root}", flush=True)


if __name__ == "__main__":
    main()
