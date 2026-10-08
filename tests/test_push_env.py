"""Tests for C3.5: PushTrack-v0 gymnasium environment.

Run:  python -m pytest tests/test_push_env.py -v
"""
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
