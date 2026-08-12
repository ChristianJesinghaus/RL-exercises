from __future__ import annotations

import json

import numpy as np
import torch

from rl_exercises.final_project.envs import FixedFourRoomsEnv, make_env
from rl_exercises.final_project.experiment import (
    ExperimentConfig,
    run_experiment,
)
from rl_exercises.final_project.intrinsic import RNDModule
from rl_exercises.final_project.networks import ReplayBuffer
from rl_exercises.final_project.sweep import preset_configs


def test_observation_is_partial_and_position_is_logging_only():
    env = make_env(seed=4, noise_dim=16)
    observation, info = env.reset(seed=4)
    expected_dim = 7 * 7 * 3 + 4 + 16
    assert observation.shape == (expected_dim,)
    assert observation.dtype == np.float32
    assert observation.min() >= 0.0
    assert observation.max() <= 1.0
    assert "agent_position" in info
    assert "agent_direction" in info
    # Position is deliberately absent from the model input; only image,
    # direction, and the TV vector account for every input dimension.
    assert expected_dim == env.observation_dim
    env.close()


def test_tv_noise_is_seeded_and_clean_variant_is_zero():
    noisy = make_env(noisy_tv=True, seed=12, noise_dim=8)
    first, info = noisy.reset(seed=12)
    repeated, _ = noisy.reset(seed=12)
    assert info["in_tv_zone"]
    np.testing.assert_array_equal(first, repeated)
    assert np.any(first[-8:] != 0.0)
    changed, *_ = noisy.step(0)
    assert not np.array_equal(first[-8:], changed[-8:])
    noisy.close()

    clean = make_env(noisy_tv=False, seed=12, noise_dim=8)
    observation, _ = clean.reset(seed=12)
    np.testing.assert_array_equal(observation[-8:], np.zeros(8, dtype=np.float32))
    clean.close()


def test_fixed_layout_and_controlled_route_reach_goal():
    env = make_env(noisy_tv=False, seed=0)
    env.reset(seed=0)
    assert len(env.unwrapped.reachable_positions) == 68
    assert env.unwrapped.grid.get(5, 3).type == "wall"
    assert env.unwrapped.grid.get(5, 2) is None

    terminated = False
    reward = 0.0
    for _ in range(7):  # east through the first opening
        _, reward, terminated, _, _ = env.step(2)
    env.step(1)  # face south
    for _ in range(6):  # south through the second opening to the goal
        _, reward, terminated, _, _ = env.step(2)
    assert terminated
    assert reward > 0.0
    assert tuple(env.unwrapped.agent_pos) == FixedFourRoomsEnv.goal_pos
    env.close()


def test_replay_buffer_shapes_and_sampling():
    buffer = ReplayBuffer(capacity=20, observation_dim=5, seed=1)
    for index in range(10):
        observation = np.full(5, index, dtype=np.float32)
        buffer.add(observation, index % 3, float(index), observation + 1, False)
    batch = buffer.sample(4, torch.device("cpu"))
    assert batch.observations.shape == (4, 5)
    assert batch.next_observations.shape == (4, 5)
    assert batch.actions.shape == (4,)
    assert batch.rewards.shape == (4,)


def test_rnd_target_remains_frozen_after_predictor_update():
    torch.manual_seed(2)
    module = RNDModule(10, mode="rnd", learning_rate=1e-3)
    before = {
        name: parameter.detach().clone()
        for name, parameter in module.target.named_parameters()
    }
    observations = torch.rand(32, 10)
    for _ in range(5):
        module.update(observations)
    for name, parameter in module.target.named_parameters():
        torch.testing.assert_close(parameter, before[name])
        assert not parameter.requires_grad


def test_lp_rnd_is_nonnegative_and_snapshot_resets_progress():
    torch.manual_seed(3)
    module = RNDModule(
        10,
        mode="lp_rnd",
        learning_rate=1e-3,
        snapshot_interval=1,
    )
    observations = torch.rand(32, 10)
    initial = module.signals(observations[0].numpy())["intrinsic_raw"]
    assert initial == 0.0
    for _ in range(50):
        module.update(observations)
    learned = module.signals(observations[0].numpy())["intrinsic_raw"]
    assert learned > 0.0
    assert module.maybe_refresh_lagged_predictor(1)
    refreshed = module.signals(observations[0].numpy())["intrinsic_raw"]
    assert refreshed >= 0.0
    assert refreshed < 1e-8


def test_preset_run_counts_and_beta_matrix(tmp_path):
    smoke = preset_configs("smoke", tmp_path / "smoke")
    pilot = preset_configs("pilot", tmp_path / "pilot")
    main = preset_configs("main", tmp_path / "main")
    assert len(smoke) == 3
    assert len(pilot) == 9
    assert len(main) == 35
    assert {config.beta for config in main if config.method != "dqn"} == {
        0.01,
        0.05,
        0.1,
    }


def test_short_lp_rnd_run_writes_complete_artifacts(tmp_path):
    config = ExperimentConfig(
        method="lp_rnd",
        seed=5,
        total_steps=80,
        beta=0.01,
        output_dir=str(tmp_path / "run"),
        learning_starts=32,
        batch_size=32,
        buffer_capacity=200,
        target_update_interval=40,
        epsilon_decay_steps=80,
        lp_snapshot_interval=40,
        eval_interval=40,
        eval_episodes=1,
        log_interval=20,
    )
    summary = run_experiment(config)
    assert summary["status"] == "complete"
    for filename in (
        "config.json",
        "run_state.json",
        "summary.json",
        "episodes.csv",
        "evaluations.csv",
        "diagnostics.csv",
        "visitation.npy",
        "checkpoint.pt",
    ):
        assert (tmp_path / "run" / filename).exists()
    state = json.loads(
        (tmp_path / "run" / "run_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
