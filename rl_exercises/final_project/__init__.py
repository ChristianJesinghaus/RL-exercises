"""Final project: noise-robust intrinsic motivation in MiniGrid."""

from rl_exercises.final_project.envs import (
    FixedFourRoomsEnv,
    NoisyTVObservationWrapper,
    make_env,
)

__all__ = ["FixedFourRoomsEnv", "NoisyTVObservationWrapper", "make_env"]
