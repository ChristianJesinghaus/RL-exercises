"""Random Network Distillation and lagged-predictor learning progress.
ChatGPT 5.5 Thinking was used for parts of code production, review and overall help in this file.

"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from rl_exercises.final_project.networks import RNDNetwork

IntrinsicMode = Literal["rnd", "lp_rnd"]


@dataclass
class RunningMoments:
    """Numerically stable scalar running mean and variance."""

    count: float = 1e-4
    mean: float = 0.0
    m2: float = 1e-4

    @property
    def variance(self) -> float:
        return max(self.m2 / self.count, 1e-8)

    def update(self, values: np.ndarray | float) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        batch_count = float(array.size)
        batch_mean = float(array.mean())
        batch_m2 = float(((array - batch_mean) ** 2).sum())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.count = total

    def normalize(
        self,
        value: float,
        *,
        update: bool = True,
        clip: float = 5.0,
    ) -> float:
        if update:
            self.update(value)
        # Intrinsic rewards stay non-negative; only scale by the running std.
        normalized = float(value) / float(np.sqrt(self.variance) + 1e-8)
        return float(np.clip(normalized, 0.0, clip))


class RNDModule:
    """RND bonus with optional lagged-predictor learning-progress gating."""

    def __init__(
        self,
        observation_dim: int,
        *,
        mode: IntrinsicMode,
        hidden_dim: int = 128,
        feature_dim: int = 64,
        learning_rate: float = 1e-4,
        snapshot_interval: int = 1_000,
        bonus_clip: float = 5.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if mode not in ("rnd", "lp_rnd"):
            raise ValueError(f"Unsupported intrinsic mode: {mode}")
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be positive")

        self.mode = mode
        self.device = torch.device(device)
        self.snapshot_interval = int(snapshot_interval)
        self.bonus_clip = float(bonus_clip)

        self.target = RNDNetwork(observation_dim, hidden_dim, feature_dim).to(
            self.device
        )
        self.predictor = RNDNetwork(observation_dim, hidden_dim, feature_dim).to(
            self.device
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()

        self.lagged_predictor: nn.Module | None = None
        if mode == "lp_rnd":
            self.lagged_predictor = copy.deepcopy(self.predictor).to(self.device)
            for parameter in self.lagged_predictor.parameters():
                parameter.requires_grad_(False)
            self.lagged_predictor.eval()

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.moments = RunningMoments()
        self.last_snapshot_step = 0

    @staticmethod
    def _per_sample_error(
        prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean((prediction - target) ** 2, dim=-1)

    def signals(self, observation: np.ndarray | torch.Tensor) -> dict[str, float]:
        """Return current RND error and the selected raw intrinsic signal."""

        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            target_features = self.target(tensor)
            current_features = self.predictor(tensor)
            current_error = self._per_sample_error(current_features, target_features)

            if self.mode == "rnd":
                raw_bonus = current_error
                old_error = current_error
            else:
                assert self.lagged_predictor is not None
                old_features = self.lagged_predictor(tensor)
                old_error = self._per_sample_error(old_features, target_features)
                raw_bonus = torch.clamp(old_error - current_error, min=0.0)

        return {
            "rnd_error": float(current_error.mean().item()),
            "old_rnd_error": float(old_error.mean().item()),
            "intrinsic_raw": float(raw_bonus.mean().item()),
        }

    def normalized_bonus(
        self,
        observation: np.ndarray | torch.Tensor,
        *,
        update_moments: bool = True,
    ) -> dict[str, float]:
        signals = self.signals(observation)
        signals["intrinsic_normalized"] = self.moments.normalize(
            signals["intrinsic_raw"],
            update=update_moments,
            clip=self.bonus_clip,
        )
        return signals

    def update(self, observations: torch.Tensor) -> float:
        """Train the current predictor while keeping the target frozen."""

        with torch.no_grad():
            targets = self.target(observations)
        predictions = self.predictor(observations)
        loss = torch.mean((predictions - targets) ** 2)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.detach().item())

    def maybe_refresh_lagged_predictor(self, step: int) -> bool:
        if self.mode != "lp_rnd" or step % self.snapshot_interval != 0:
            return False
        assert self.lagged_predictor is not None
        self.lagged_predictor.load_state_dict(self.predictor.state_dict())
        self.last_snapshot_step = int(step)
        return True

    def checkpoint(self) -> dict:
        state = {
            "mode": self.mode,
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "moments": asdict(self.moments),
            "last_snapshot_step": self.last_snapshot_step,
        }
        if self.lagged_predictor is not None:
            state["lagged_predictor"] = self.lagged_predictor.state_dict()
        return state
