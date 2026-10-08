"""Tests for C3.5: PushTrack-v0 gymnasium environment.

Run:  python -m pytest tests/test_push_env.py -v
"""
import json
import pathlib
import time

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import sim.push_env  # registers PushTrack-v0


def _make_env(**kwargs):
    return gym.make("PushTrack-v0", **kwargs)


def test_check_env():
    """gymnasium check_env must pass with no warnings or errors."""
    env = PushTrackEnv_direct()
    check_env(env, warn=True)
    env.close()


def PushTrackEnv_direct():
    """Return a raw PushTrackEnv (not wrapped) for check_env."""
    from sim.push_env import PushTrackEnv
    return PushTrackEnv()


def test_seeded_reset_reproducible():
    """Two resets with the same seed must return identical observations."""
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv()
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    assert np.allclose(obs1, obs2, atol=1e-6), (
        "Seeded reset not reproducible: observations differ"
    )
    env.close()


def test_obs_shape_and_dtype():
    """Observation shape and dtype match the declared space."""
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(k_ref=5)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape, (
        f"obs shape {obs.shape} != space shape {env.observation_space.shape}"
    )
    assert obs.dtype == np.float32, f"obs dtype {obs.dtype}"
    env.close()


def test_step_returns_correct_types():
    """step() must return (obs, reward, terminated, truncated, info) with correct types."""
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv()
    env.reset(seed=1)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert "success" in info
    env.close()


def test_env_throughput():
    """Report env steps/sec (no threshold; informational for RL feasibility).

    Typical target for RL training: > 200 env steps/sec.
    """
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(max_episode_steps=50)
    env.reset(seed=7)

    n_steps = 200
    t0 = time.perf_counter()
    done_count = 0
    step_count = 0
    for _ in range(n_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        step_count += 1
        if terminated or truncated:
            env.reset(seed=step_count)
            done_count += 1
    elapsed = time.perf_counter() - t0
    rate = step_count / elapsed

    print(f"\nPushTrackEnv throughput: {rate:.0f} steps/sec "
          f"({step_count} steps, {done_count} episodes in {elapsed:.2f} s)")
    assert rate > 0, "throughput must be positive"
    env.close()


# ---------------------------------------------------------------------------
# split= episodes (recorded data through the stored WorkspaceMap)
# ---------------------------------------------------------------------------

_SPLITS = pathlib.Path("data/splits.json")
_MAP = pathlib.Path("data/workspace_map.json")

needs_data = pytest.mark.skipif(
    not (_SPLITS.exists() and _MAP.exists() and pathlib.Path("data/processed").exists()),
    reason="needs data/splits.json, data/workspace_map.json and data/processed/",
)


def test_resample_episode():
    """Grid-aligned input is unchanged; a 10 Hz input is interpolated, yaw across ±pi too."""
    from sim.push_env import resample_episode
    t = np.arange(5) * 0.05
    pose = np.column_stack([t, 2 * t, np.full(5, 0.3)])
    assert np.allclose(resample_episode(t, pose, 0.05), pose)

    t2 = np.array([0.0, 0.1, 0.2])
    pose2 = np.array([[0.0, 0.0, np.pi - 0.1], [1.0, 0.0, -np.pi + 0.1], [2.0, 0.0, -np.pi + 0.3]])
    out = resample_episode(t2, pose2, 0.05)
    assert out.shape == (5, 3)
    assert np.allclose(out[1, :2], [0.5, 0.0])
    assert abs(abs(out[1, 2]) - np.pi) < 1e-9     # midway through the wrap, not through 0


def test_paths_and_split_are_exclusive():
    from sim.push_env import PushTrackEnv
    with pytest.raises(ValueError):
        PushTrackEnv(paths=[np.zeros((3, 2))], split="train")


@needs_data
def test_split_train_reset_reproducible_and_start_matches_mapped_sample():
    """Seeded reset on split='train' repeats, and the cube starts at the mapped first sample."""
    from control.workspace import load_workspace_map
    from sim.push_env import PushTrackEnv

    with open(_SPLITS) as f:
        train = json.load(f)["train"]
    ws_map = load_workspace_map()

    env = PushTrackEnv(split="train")
    assert env._episode_ids == train            # only train episodes are loaded

    obs1, info1 = env.reset(seed=3)
    obs2, info2 = env.reset(seed=3)
    assert info1 == info2 and info1["episode_id"] in train
    assert np.allclose(obs1, obs2, atol=1e-6)

    d = np.load(pathlib.Path("data/processed") / f"{info1['episode_id']}.npz")
    x, y, yaw = d["cube_xy_yaw"][0]
    expected_xy = ws_map.transform(np.array([x, y]))
    cube_xy, cube_yaw = env._cube_xy_yaw()
    assert np.allclose(cube_xy, expected_xy, atol=1e-9)
    assert abs(np.arctan2(np.sin(cube_yaw - yaw), np.cos(cube_yaw - yaw))) < 1e-9
    env.close()


@needs_data
def test_split_reset_can_select_episode():
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(split="showcase")
    _, info = env.reset(seed=0, options={"episode": 1})
    assert info["episode_id"] == env._episode_ids[1]
    env.close()


@needs_data
def test_episode_files_load_like_split_episodes():
    """An explicit .npz is mapped, resampled and started exactly as the same episode in its split."""
    from sim.push_env import PushTrackEnv
    ref = PushTrackEnv(split="test", episodes=["ep_005"])
    env = PushTrackEnv(episode_files=[pathlib.Path("data/processed/ep_005.npz")])
    assert env.episode_ids == ["ep_005"]
    _, info = env.reset(options={"episode_id": "ep_005"})
    ref.reset(options={"episode_id": "ep_005"})
    assert info["episode_id"] == "ep_005"
    assert np.array_equal(env.reference_path, ref.reference_path)
    assert env.cube_xy_yaw[1] == ref.cube_xy_yaw[1]
    with pytest.raises(ValueError):
        PushTrackEnv(split="test", episode_files=[pathlib.Path("data/processed/ep_005.npz")])
    env.close()
    ref.close()
