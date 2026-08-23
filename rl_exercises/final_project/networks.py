"""Neural networks and replay buffer used by the final-project agents.
ChatGPT 5.5 Thinking was used for parts of code production, review and overall help in this file.

"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_layers: int = 2,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(hidden_layers):
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Small feed-forward action-value network."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.model = _mlp(observation_dim, hidden_dim, action_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


class RNDNetwork(nn.Module):
    """MLP used as either the fixed target or trainable RND predictor."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int = 128,
        feature_dim: int = 64,
    ) -> None:
        super().__init__()
        self.model = _mlp(observation_dim, hidden_dim, feature_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


@dataclass(frozen=True)
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Fixed-size NumPy replay buffer with deterministic sampling."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        seed: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self._index = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminated: bool,
    ) -> None:
        index = self._index
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = float(terminated)
        self._index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {self._size}"
            )
        indices = self._rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            observations=torch.as_tensor(
                self.observations[indices], dtype=torch.float32, device=device
            ),
            actions=torch.as_tensor(
                self.actions[indices], dtype=torch.long, device=device
            ),
            rewards=torch.as_tensor(
                self.rewards[indices], dtype=torch.float32, device=device
            ),
            next_observations=torch.as_tensor(
                self.next_observations[indices],
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                self.terminated[indices], dtype=torch.float32, device=device
            ),
        )
