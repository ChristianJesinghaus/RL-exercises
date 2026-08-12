"""Train and evaluate DQN, DQN+RND, or DQN+LP-RND.

Examples
--------
python -m rl_exercises.final_project.experiment \
    --method lp_rnd --seed 0 --beta 0.01 --total-steps 50000 \
    --output-dir results/lp_rnd_seed0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from rl_exercises.final_project.envs import make_env
from rl_exercises.final_project.intrinsic import RNDModule
from rl_exercises.final_project.networks import QNetwork, ReplayBatch, ReplayBuffer

Method = Literal["dqn", "rnd", "lp_rnd"]


@dataclass
class ExperimentConfig:
    """All settings required for one reproducible run."""

    method: Method = "lp_rnd"
    seed: int = 0
    total_steps: int = 50_000
    beta: float = 0.01
    output_dir: str = "results/single_run"

    noisy_tv: bool = True
    noise_probability: float = 1.0
    noise_dim: int = 16
    max_episode_steps: int = 200

    gamma: float = 0.99
    learning_rate: float = 1e-4
    hidden_dim: int = 128
    batch_size: int = 64
    buffer_capacity: int = 50_000
    learning_starts: int = 1_000
    train_frequency: int = 1
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    rnd_learning_rate: float = 1e-4
    rnd_hidden_dim: int = 128
    rnd_feature_dim: int = 64
    lp_snapshot_interval: int = 1_000
    intrinsic_clip: float = 5.0

    eval_interval: int = 5_000
    eval_episodes: int = 20
    log_interval: int = 1_000
    torch_threads: int = 1
    device: str = "cpu"

    def validate(self) -> None:
        if self.method not in ("dqn", "rnd", "lp_rnd"):
            raise ValueError(f"Unknown method: {self.method}")
        if self.total_steps < 1:
            raise ValueError("total_steps must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not 0.0 <= self.noise_probability <= 1.0:
            raise ValueError("noise_probability must be between 0 and 1")
        if self.batch_size < 1 or self.buffer_capacity < self.batch_size:
            raise ValueError("buffer_capacity must be at least batch_size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        for name in (
            "train_frequency",
            "target_update_interval",
            "epsilon_decay_steps",
            "eval_interval",
            "eval_episodes",
            "log_interval",
            "torch_threads",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.device != "cpu" and not torch.cuda.is_available():
            raise ValueError(
                f"Requested device {self.device!r}, but CUDA is unavailable"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid_names = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid_names)
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(unknown)}")
        return cls(**data)


class CSVStream:
    """Append rows and flush immediately so interrupted runs retain data."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.handle = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class DiagnosticTracker:
    """Track analysis-only state and binned intrinsic-reward diagnostics."""

    _signal_names = ("rnd_error", "intrinsic_raw", "intrinsic_normalized")

    def __init__(self, grid_size: int, reachable_count: int) -> None:
        self.visitation = np.zeros((grid_size, grid_size), dtype=np.int64)
        self.reachable_count = reachable_count
        self.positions: set[tuple[int, int]] = set()
        self.abstract_states: set[tuple[int, int, int]] = set()
        self.observation_hashes: set[str] = set()
        self.total_steps = 0
        self.total_tv_steps = 0
        self._reset_block()

    def _reset_block(self) -> None:
        self.block_steps = 0
        self.block_tv_steps = 0
        self.signal_counts = {"inside": 0, "outside": 0}
        self.signal_sums = {
            (zone, signal): 0.0
            for zone in ("inside", "outside")
            for signal in self._signal_names
        }
        self.q_losses: list[float] = []
        self.rnd_losses: list[float] = []

    def observe(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        signals: dict[str, float],
    ) -> None:
        x, y = (int(value) for value in info["agent_position"])
        direction = int(info["agent_direction"])
        in_tv_zone = bool(info["in_tv_zone"])

        self.visitation[y, x] += 1
        self.positions.add((x, y))
        self.abstract_states.add((x, y, direction))
        digest = hashlib.blake2b(observation.tobytes(), digest_size=8).hexdigest()
        self.observation_hashes.add(digest)

        self.total_steps += 1
        self.block_steps += 1
        if in_tv_zone:
            self.total_tv_steps += 1
            self.block_tv_steps += 1

        zone = "inside" if in_tv_zone else "outside"
        rnd_error = signals["rnd_error"]
        if np.isfinite(rnd_error):
            self.signal_counts[zone] += 1
            for signal_name in self._signal_names:
                self.signal_sums[(zone, signal_name)] += float(signals[signal_name])

    def add_losses(self, q_loss: float, rnd_loss: float | None) -> None:
        self.q_losses.append(float(q_loss))
        if rnd_loss is not None:
            self.rnd_losses.append(float(rnd_loss))

    def _zone_mean(self, zone: str, signal: str) -> float:
        count = self.signal_counts[zone]
        if count == 0:
            return float("nan")
        return self.signal_sums[(zone, signal)] / count

    def row(self, step: int) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "step": step,
            "tv_fraction_block": self.block_tv_steps / max(self.block_steps, 1),
            "tv_fraction_cumulative": self.total_tv_steps / max(self.total_steps, 1),
            "position_coverage": len(self.positions) / self.reachable_count,
            "unique_positions": len(self.positions),
            "unique_abstract_states": len(self.abstract_states),
            "unique_observations": len(self.observation_hashes),
            "rnd_error_inside": self._zone_mean("inside", "rnd_error"),
            "rnd_error_outside": self._zone_mean("outside", "rnd_error"),
            "intrinsic_raw_inside": self._zone_mean("inside", "intrinsic_raw"),
            "intrinsic_raw_outside": self._zone_mean("outside", "intrinsic_raw"),
            "intrinsic_normalized_inside": self._zone_mean(
                "inside", "intrinsic_normalized"
            ),
            "intrinsic_normalized_outside": self._zone_mean(
                "outside", "intrinsic_normalized"
            ),
            "mean_q_loss": (
                float(np.mean(self.q_losses)) if self.q_losses else float("nan")
            ),
            "mean_rnd_loss": (
                float(np.mean(self.rnd_losses)) if self.rnd_losses else float("nan")
            ),
        }
        self._reset_block()
        return row


EPISODE_FIELDS = [
    "episode",
    "end_step",
    "length",
    "extrinsic_return",
    "shaped_return",
    "success",
    "tv_fraction",
]
EVALUATION_FIELDS = [
    "step",
    "mean_extrinsic_return",
    "success_rate",
    "mean_episode_length",
    "tv_fraction",
]
DIAGNOSTIC_FIELDS = [
    "step",
    "tv_fraction_block",
    "tv_fraction_cumulative",
    "position_coverage",
    "unique_positions",
    "unique_abstract_states",
    "unique_observations",
    "rnd_error_inside",
    "rnd_error_outside",
    "intrinsic_raw_inside",
    "intrinsic_raw_outside",
    "intrinsic_normalized_inside",
    "intrinsic_normalized_outside",
    "mean_q_loss",
    "mean_rnd_loss",
]


def set_global_seeds(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(torch_threads)


def epsilon_at_step(config: ExperimentConfig, step: int) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def double_dqn_update(
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    batch: ReplayBatch,
    gamma: float,
) -> float:
    q_values = (
        q_network(batch.observations).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
    )
    with torch.no_grad():
        next_actions = q_network(batch.next_observations).argmax(dim=1, keepdim=True)
        next_q_values = (
            target_network(batch.next_observations).gather(1, next_actions).squeeze(1)
        )
        targets = batch.rewards + gamma * (1.0 - batch.terminated) * next_q_values

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.detach().item())


def greedy_action(
    q_network: QNetwork, observation: np.ndarray, device: torch.device
) -> int:
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device
        ).unsqueeze(0)
        return int(q_network(tensor).argmax(dim=1).item())


def evaluate(
    q_network: QNetwork,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate greedily with extrinsic reward only."""

    env = make_env(
        noisy_tv=config.noisy_tv,
        noise_probability=config.noise_probability,
        noise_dim=config.noise_dim,
        max_steps=config.max_episode_steps,
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes = 0
    tv_steps = 0
    total_steps = 0
    was_training = q_network.training
    q_network.eval()

    try:
        for episode in range(config.eval_episodes):
            observation, info = env.reset(seed=config.seed + 1_000_000 + episode)
            episode_return = 0.0
            for length in range(1, config.max_episode_steps + 1):
                action = greedy_action(q_network, observation, device)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                total_steps += 1
                tv_steps += int(info["in_tv_zone"])
                if terminated or truncated:
                    successes += int(terminated and reward > 0.0)
                    break
            returns.append(episode_return)
            lengths.append(length)
    finally:
        env.close()
        q_network.train(was_training)

    return {
        "mean_extrinsic_return": float(np.mean(returns)),
        "success_rate": successes / config.eval_episodes,
        "mean_episode_length": float(np.mean(lengths)),
        "tv_fraction": tv_steps / max(total_steps, 1),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _success_auc(evaluations: list[dict[str, float]]) -> float:
    if len(evaluations) < 2:
        return 0.0
    area = 0.0
    for previous, current in pairwise(evaluations):
        width = current["step"] - previous["step"]
        height = 0.5 * (previous["success_rate"] + current["success_rate"])
        area += width * height
    final_step = evaluations[-1]["step"]
    return float(area / max(final_step, 1.0))


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    config: ExperimentConfig,
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    intrinsic: RNDModule | None,
) -> None:
    payload: dict[str, Any] = {
        "step": step,
        "config": asdict(config),
        "q_network": q_network.state_dict(),
        "target_network": target_network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if intrinsic is not None:
        payload["intrinsic"] = intrinsic.checkpoint()
    torch.save(payload, path)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute one complete training run and return its summary."""

    config.validate()
    output_dir = Path(config.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new directory to protect existing results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seeds(config.seed, config.torch_threads)
    device = torch.device(config.device)
    _write_json(output_dir / "config.json", asdict(config))
    _write_json(
        output_dir / "run_state.json",
        {"status": "running", "started_unix": time.time()},
    )

    env = make_env(
        noisy_tv=config.noisy_tv,
        noise_probability=config.noise_probability,
        noise_dim=config.noise_dim,
        max_steps=config.max_episode_steps,
    )
    observation, _ = env.reset(seed=config.seed)
    observation_dim = int(observation.shape[0])
    action_dim = int(env.action_space.n)
    reachable_count = len(env.unwrapped.reachable_positions)

    q_network = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    target_network = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(q_network.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.buffer_capacity, observation_dim, config.seed + 17)

    intrinsic: RNDModule | None = None
    if config.method in ("rnd", "lp_rnd"):
        intrinsic = RNDModule(
            observation_dim,
            mode=config.method,
            hidden_dim=config.rnd_hidden_dim,
            feature_dim=config.rnd_feature_dim,
            learning_rate=config.rnd_learning_rate,
            snapshot_interval=config.lp_snapshot_interval,
            bonus_clip=config.intrinsic_clip,
            device=device,
        )

    diagnostics = DiagnosticTracker(env.unwrapped.size, reachable_count)
    policy_rng = np.random.default_rng(config.seed + 23)
    episode_stream = CSVStream(output_dir / "episodes.csv", EPISODE_FIELDS)
    evaluation_stream = CSVStream(output_dir / "evaluations.csv", EVALUATION_FIELDS)
    diagnostic_stream = CSVStream(output_dir / "diagnostics.csv", DIAGNOSTIC_FIELDS)

    episode_index = 0
    episode_length = 0
    episode_extrinsic_return = 0.0
    episode_shaped_return = 0.0
    episode_tv_steps = 0
    reward_discoveries: list[int] = []
    evaluation_rows: list[dict[str, float]] = []
    start_time = time.perf_counter()
    last_log_step = 0
    last_eval_step = -1

    def record_evaluation(step: int) -> None:
        nonlocal last_eval_step
        values = evaluate(q_network, config, device)
        row = {"step": step, **values}
        evaluation_stream.write(row)
        evaluation_rows.append(row)
        _save_checkpoint(
            output_dir / "checkpoint.pt",
            step=step,
            config=config,
            q_network=q_network,
            target_network=target_network,
            optimizer=optimizer,
            intrinsic=intrinsic,
        )
        last_eval_step = step

    try:
        record_evaluation(0)
        for step in range(1, config.total_steps + 1):
            epsilon = epsilon_at_step(config, step)
            if policy_rng.random() < epsilon:
                action = int(policy_rng.integers(action_dim))
            else:
                action = greedy_action(q_network, observation, device)

            (
                next_observation,
                extrinsic_reward,
                terminated,
                truncated,
                next_info,
            ) = env.step(action)

            if intrinsic is None:
                signals = {
                    "rnd_error": float("nan"),
                    "old_rnd_error": float("nan"),
                    "intrinsic_raw": 0.0,
                    "intrinsic_normalized": 0.0,
                }
            else:
                signals = intrinsic.normalized_bonus(next_observation)

            shaped_reward = (
                extrinsic_reward + config.beta * signals["intrinsic_normalized"]
            )
            replay.add(
                observation,
                action,
                shaped_reward,
                next_observation,
                terminated,
            )
            diagnostics.observe(next_observation, next_info, signals)

            episode_length += 1
            episode_extrinsic_return += extrinsic_reward
            episode_shaped_return += shaped_reward
            episode_tv_steps += int(next_info["in_tv_zone"])
            if extrinsic_reward > 0.0:
                reward_discoveries.append(step)

            if (
                len(replay) >= config.learning_starts
                and step % config.train_frequency == 0
            ):
                batch = replay.sample(config.batch_size, device)
                q_loss = double_dqn_update(
                    q_network,
                    target_network,
                    optimizer,
                    batch,
                    config.gamma,
                )
                rnd_loss = (
                    intrinsic.update(batch.next_observations)
                    if intrinsic is not None
                    else None
                )
                diagnostics.add_losses(q_loss, rnd_loss)
                if intrinsic is not None:
                    intrinsic.maybe_refresh_lagged_predictor(step)

            if step % config.target_update_interval == 0:
                target_network.load_state_dict(q_network.state_dict())

            if terminated or truncated:
                episode_stream.write(
                    {
                        "episode": episode_index,
                        "end_step": step,
                        "length": episode_length,
                        "extrinsic_return": episode_extrinsic_return,
                        "shaped_return": episode_shaped_return,
                        "success": int(terminated and episode_extrinsic_return > 0.0),
                        "tv_fraction": episode_tv_steps / max(episode_length, 1),
                    }
                )
                episode_index += 1
                episode_length = 0
                episode_extrinsic_return = 0.0
                episode_shaped_return = 0.0
                episode_tv_steps = 0
                observation, _ = env.reset()
            else:
                observation = next_observation

            if step % config.log_interval == 0:
                diagnostic_stream.write(diagnostics.row(step))
                np.save(output_dir / "visitation.npy", diagnostics.visitation)
                last_log_step = step
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{config.method} seed={config.seed}] "
                    f"step {step}/{config.total_steps} "
                    f"epsilon={epsilon:.3f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if step % config.eval_interval == 0:
                record_evaluation(step)

        if last_log_step != config.total_steps:
            diagnostic_stream.write(diagnostics.row(config.total_steps))
            np.save(output_dir / "visitation.npy", diagnostics.visitation)
        if last_eval_step != config.total_steps:
            record_evaluation(config.total_steps)

        final_eval = evaluation_rows[-1]
        summary: dict[str, Any] = {
            "status": "complete",
            "method": config.method,
            "seed": config.seed,
            "beta": 0.0 if config.method == "dqn" else config.beta,
            "total_steps": config.total_steps,
            "final_eval_success_rate": final_eval["success_rate"],
            "best_eval_success_rate": max(
                row["success_rate"] for row in evaluation_rows
            ),
            "final_eval_extrinsic_return": final_eval["mean_extrinsic_return"],
            "success_rate_auc": _success_auc(evaluation_rows),
            "first_reward_step": (
                reward_discoveries[0] if reward_discoveries else None
            ),
            "third_reward_step": (
                reward_discoveries[2] if len(reward_discoveries) >= 3 else None
            ),
            "training_successes": len(reward_discoveries),
            "training_tv_fraction": diagnostics.total_tv_steps
            / max(diagnostics.total_steps, 1),
            "position_coverage": len(diagnostics.positions) / reachable_count,
            "unique_positions": len(diagnostics.positions),
            "unique_abstract_states": len(diagnostics.abstract_states),
            "unique_observations": len(diagnostics.observation_hashes),
            "elapsed_seconds": time.perf_counter() - start_time,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(
            output_dir / "run_state.json",
            {"status": "complete", "finished_unix": time.time()},
        )
        return summary
    except BaseException as error:
        _write_json(
            output_dir / "run_state.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_unix": time.time(),
            },
        )
        raise
    finally:
        episode_stream.close()
        evaluation_stream.close()
        diagnostic_stream.close()
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-json",
        type=str,
        help="Load every setting from an ExperimentConfig JSON file.",
    )
    parser.add_argument("--method", choices=("dqn", "rnd", "lp_rnd"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--noise-probability", type=float)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--buffer-capacity", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--torch-threads", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = (
        ExperimentConfig.from_json(args.config_json)
        if args.config_json
        else ExperimentConfig()
    )
    updates = {
        "method": args.method,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "beta": args.beta,
        "output_dir": args.output_dir,
        "noise_probability": args.noise_probability,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "log_interval": args.log_interval,
        "learning_starts": args.learning_starts,
        "buffer_capacity": args.buffer_capacity,
        "batch_size": args.batch_size,
        "torch_threads": args.torch_threads,
    }
    for key, value in updates.items():
        if value is not None:
            setattr(config, key, value)
    if args.clean:
        config.noisy_tv = False
    if not args.config_json:
        required = ("method", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(
                "Missing arguments without --config-json: " + ", ".join(missing)
            )
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
