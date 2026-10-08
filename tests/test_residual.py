"""C6.5 residual wrapper and controller: zero residual is the scripted pusher, the residual is bounded."""
import numpy as np
import pytest

from control.scripted_pusher import ScriptedPusher
from eval.report import rollout
from rl.residual import RESIDUAL_SCALE, ResidualActions, ResidualController
from sim.push_env import PushTrackEnv

EP = "ep_010"      # 12 cm straight push, train split
STEPS = 60


class _ZeroModel:
    def predict(self, obs, deterministic=True):
        return np.zeros((len(obs), 2), np.float32), None


def _env(max_episode_steps=600):
    return PushTrackEnv(split="train", episodes=[EP], max_episode_steps=max_episode_steps)


def test_zero_residual_executes_the_standalone_scripted_actions_step_by_step():
    env = _env()
    obs, _ = env.reset(seed=0)
    pusher = ScriptedPusher()
    pusher.reset(env.reference_path)
    expected, expected_obs = [], []
    for _ in range(STEPS):
        a = pusher(obs)
        expected.append(a)
        obs, _, term, trunc, _ = env.step(a)
        expected_obs.append(obs)
        if term or trunc:
            break
    env.close()

    w = ResidualActions(_env())
    wobs, _ = w.reset(seed=0)
    got, got_obs = [], []
    for _ in range(len(expected)):
        got.append(w._base_action.copy())
        wobs, _, term, trunc, _ = w.step(np.zeros(2))
        got_obs.append(wobs[:-2])
    w.close()
    assert np.allclose(got, expected, atol=1e-12)
    assert np.allclose(got_obs, expected_obs, atol=1e-6)


def test_observation_is_env_obs_plus_base_action_over_max_delta():
    w = ResidualActions(_env())
    obs, _ = w.reset(seed=0)
    n = w.unwrapped.observation_space.shape[0]
    assert obs.shape == (n + 2,) == w.observation_space.shape and obs.dtype == np.float32
    assert np.allclose(obs[:n], w.unwrapped._get_obs())
    assert np.allclose(obs[n:], w._base_action / w.max_delta)
    obs, *_ = w.step(np.zeros(2))
    assert np.allclose(obs[n:], w._base_action / w.max_delta)
    assert np.max(np.abs(obs[n:])) <= 1.0 + 1e-6
    w.close()


def test_residual_is_bounded_by_scale_times_max_delta():
    w = ResidualActions(_env())
    assert w.residual_scale == RESIDUAL_SCALE == 0.25
    sent = []
    w.base.executed = lambda cmd: sent.append(np.array(cmd))      # the command the stuck detection is told
    w.reset(seed=0)
    rng = np.random.default_rng(0)
    for i in range(30):
        base = w._base_action.copy()
        a = [5.0, -5.0] if i == 0 else rng.uniform(-3, 3, 2)      # out-of-range actions are clipped to [-1, 1]
        w.step(a)
        assert np.all(np.abs(sent[-1] - base) <= w.residual_scale * w.max_delta + 1e-12)
        assert np.all(np.abs(sent[-1]) <= w.max_delta + 1e-12)
        if i == 0:
            # a = (5, -5) acts as (1, -1): not 5x the bound, and not nothing
            assert np.max(np.abs(sent[-1] - base)) > 0.5 * w.residual_scale * w.max_delta
    w.close()


def test_stuck_detection_sees_the_executed_command():
    p = ScriptedPusher()
    p.reset([[0.0, 0.0], [1.0, 0.0]])
    p.act(np.array([0.3, 0.0, 0.0, 0.0, 0.0, 1.0]))
    p.executed([0.01, 0.0])
    assert np.allclose(p._last_cmd, [0.01, 0.0])


def test_residual_controller_with_zero_policy_matches_scripted_pusher():
    env = _env()
    ref = rollout(env, ScriptedPusher(), EP, "scripted")
    got = rollout(env, ResidualController(_ZeroModel(), None, RESIDUAL_SCALE, env.max_delta), EP, "residual")
    env.close()
    for k in ("progress", "mean_deviation_cm", "final_error_cm", "success", "steps"):
        assert got[k] == pytest.approx(ref[k]), k
    assert np.allclose(got["achieved_xy"], ref["achieved_xy"])


def test_residual_controller_done_follows_the_base_pusher():
    env = _env()
    c = ResidualController(_ZeroModel(), None, RESIDUAL_SCALE, env.max_delta)
    obs, _ = env.reset(seed=0)
    c.reset(env.reference_path)
    assert not c.done
    c(obs)
    assert c.done == c.base.done
    env.close()
