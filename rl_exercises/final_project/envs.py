"""Controlled partially observed FourRooms environment with a Noisy-TV region.

The agent, DQN, RND, and LP-RND all receive exactly the same observation:

* the standard MiniGrid 7 x 7 x 3 symbolic image,
* MiniGrid's standard egocentric direction field as a one-hot vector, and
* an irrelevant random vector that is non-zero only inside the TV region.

Global position is added to ``info`` for diagnostics only.  It is never part of
the observation returned to an agent or intrinsic-reward module.

ChatGPT 5.5 Thinking was used for parts of code production, review and overall help in this file.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv


class FixedFourRoomsEnv(MiniGridEnv):
    """An 11 x 11 FourRooms task with fixed start, doors, and goal.

    The compact geometry keeps the task representable by a feed-forward policy
    using MiniGrid's partial 7 x 7 view.  The shortest controlled route still
    crosses two room boundaries.
    """

    size = 11
    start_pos = (2, 2)
    start_dir = 0  # east
    goal_pos = (9, 8)
    wall_coordinate = 5
    openings = frozenset({(5, 2), (5, 8), (2, 5), (9, 5)})
    tv_positions = frozenset({(2, 2), (2, 3), (3, 2), (3, 3)})

    def __init__(
        self,
        max_steps: int = 200,
        render_mode: str | None = None,
        agent_view_size: int = 7,
    ) -> None:
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")
        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            see_through_walls=False,
            agent_view_size=agent_view_size,
            render_mode=render_mode,
        )
        # Only turning and moving forward are useful in this navigation task.
        self.action_space = spaces.Discrete(3)

    def _gen_grid(self, width: int, height: int) -> None:
        if width != self.size or height != self.size:
            raise ValueError(
                f"FixedFourRoomsEnv requires a {self.size} x {self.size} grid"
            )

        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        for y in range(1, height - 1):
            self.grid.set(self.wall_coordinate, y, Wall())
        for x in range(1, width - 1):
            self.grid.set(x, self.wall_coordinate, Wall())
        for opening in self.openings:
            self.grid.set(*opening, None)

        self.agent_pos = self.start_pos
        self.agent_dir = self.start_dir
        self.grid.set(*self.start_pos, None)

        goal = Goal()
        self.put_obj(goal, *self.goal_pos)
        goal.init_pos = self.goal_pos
        goal.cur_pos = self.goal_pos
        self.mission = "reach the goal"

    @property
    def reachable_positions(self) -> frozenset[tuple[int, int]]:
        """Return all non-wall positions, for coverage diagnostics only."""

        positions: set[tuple[int, int]] = set()
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if cell is None or cell.can_overlap():
                    positions.add((x, y))
        return frozenset(positions)


class NoisyTVObservationWrapper(gym.Wrapper):
    """Flatten MiniGrid observations and inject irrelevant stochastic features."""

    # MiniGrid object, color, and state indices are bounded by these values.
    _image_scales = np.asarray([10.0, 5.0, 2.0], dtype=np.float32)

    def __init__(
        self,
        env: FixedFourRoomsEnv,
        *,
        noise_probability: float = 1.0,
        noise_dim: int = 16,
        tv_positions: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__(env)
        if not 0.0 <= noise_probability <= 1.0:
            raise ValueError("noise_probability must be between 0 and 1")
        if noise_dim < 1:
            raise ValueError("noise_dim must be positive")

        self.noise_probability = float(noise_probability)
        self.noise_dim = int(noise_dim)
        self.tv_positions = frozenset(tv_positions or env.tv_positions)
        image_shape = env.observation_space["image"].shape
        self.image_dim = int(np.prod(image_shape))
        self.direction_dim = 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.image_dim + self.direction_dim + self.noise_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)
        self._noise_rng: np.random.Generator | None = None

    @property
    def observation_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def _position(self) -> tuple[int, int]:
        pos = self.env.unwrapped.agent_pos
        return int(pos[0]), int(pos[1])

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        position = self._position()
        enriched = dict(info)
        enriched.update(
            {
                "agent_position": position,
                "agent_direction": int(self.env.unwrapped.agent_dir),
                "in_tv_zone": position in self.tv_positions,
            }
        )
        return enriched

    def _transform_observation(self, observation: dict[str, Any]) -> np.ndarray:
        image = np.asarray(observation["image"], dtype=np.float32)
        image = np.clip(image / self._image_scales, 0.0, 1.0).reshape(-1)

        direction = np.zeros(self.direction_dim, dtype=np.float32)
        direction[int(observation["direction"])] = 1.0

        noise = np.zeros(self.noise_dim, dtype=np.float32)
        if self._position() in self.tv_positions:
            if self._noise_rng is None:
                self._noise_rng = np.random.default_rng()
            if self._noise_rng.random() < self.noise_probability:
                noise = self._noise_rng.random(self.noise_dim, dtype=np.float32)

        return np.concatenate((image, direction, noise), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None or self._noise_rng is None:
            # A separate stream prevents policy/environment sampling from
            # changing the distractor sequence.
            seed_sequence = np.random.SeedSequence(seed)
            self._noise_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        observation, info = self.env.reset(seed=seed, options=options)
        return self._transform_observation(observation), self._augment_info(info)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Expected movement action in [0, 2], got {action}")
        observation, reward, terminated, truncated, info = self.env.step(int(action))
        return (
            self._transform_observation(observation),
            float(reward),
            terminated,
            truncated,
            self._augment_info(info),
        )


def make_env(
    *,
    noisy_tv: bool = True,
    noise_probability: float = 1.0,
    noise_dim: int = 16,
    max_steps: int = 200,
    seed: int | None = None,
    render_mode: str | None = None,
) -> NoisyTVObservationWrapper:
    """Construct the controlled experiment environment.

    ``noisy_tv=False`` keeps the observation dimensionality unchanged but sets
    the distractor vector to zero.  ``seed`` is applied immediately so callers
    can inspect a fully initialized environment.
    """

    env = FixedFourRoomsEnv(max_steps=max_steps, render_mode=render_mode)
    wrapped = NoisyTVObservationWrapper(
        env,
        noise_probability=noise_probability if noisy_tv else 0.0,
        noise_dim=noise_dim,
    )
    if seed is not None:
        wrapped.reset(seed=seed)
    return wrapped
