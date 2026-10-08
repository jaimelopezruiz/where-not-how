"""C6.5: residual RL on the scripted pusher.

The executed EE step is the scripted pusher's step plus a small learned correction:

    step = base.act(obs) + residual_scale * max_delta * a,      a in [-1, 1]^2

so with a = 0 the system is the frozen scripted pusher (C5.2) and the policy only has to learn what to add.
The policy sees the env observation plus the base action (divided by max_delta), so it knows where the base
is about to move. The env clips the executed step to +-max_delta as usual.

ResidualActions is the training env wrapper; ResidualController is the same loop as an evaluation controller
for eval.report.rollout, with the scripted pusher's own `done` as the stopping rule (until-done protocol).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from control.scripted_pusher import ScriptedPusher

RESIDUAL_SCALE = 0.25      # the residual is at most this fraction of max_delta (5 mm at 2 cm)


def augment_obs(obs, base_action, max_delta):
    """Env observation with the base action (in units of max_delta) appended."""
    return np.concatenate([np.asarray(obs, np.float32), np.asarray(base_action, np.float32) / max_delta])


def executed_action(base_action, a, residual_scale, max_delta):
    """Base step plus the bounded residual, clipped per component as the env does."""
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    return np.clip(np.asarray(base_action, float) + residual_scale * max_delta * a, -max_delta, max_delta)


class ResidualActions(gym.Wrapper):
    """Policy-facing env: actions in [-1, 1] are residuals on a scripted pusher with default (frozen) parameters."""

    def __init__(self, env, residual_scale=RESIDUAL_SCALE):
        super().__init__(env)
        self.max_delta = float(env.unwrapped.max_delta)
        self.residual_scale = float(residual_scale)
        self.base = ScriptedPusher(max_delta=self.max_delta)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n = env.observation_space.shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n + 2,), dtype=np.float32)
        self._base_action = np.zeros(2)

    def _augment(self, obs):
        self._base_action = self.base.act(obs)
        return augment_obs(obs, self._base_action, self.max_delta)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.base.reset(self.env.unwrapped.reference_path)
        return self._augment(obs), info

    def step(self, action):
        cmd = executed_action(self._base_action, action, self.residual_scale, self.max_delta)
        self.base.executed(cmd)
        obs, reward, terminated, truncated, info = self.env.step(cmd)
        info["is_success"] = bool(info.get("success", False))
        return self._augment(obs), reward, terminated, truncated, info


class ResidualController:
    """Evaluation controller: scripted pusher + deterministic residual policy, done when the base is done."""

    def __init__(self, model, vec_normalize=None, residual_scale=RESIDUAL_SCALE, max_delta=0.02):
        self.model, self.vecnorm = model, vec_normalize
        self.residual_scale, self.max_delta = float(residual_scale), float(max_delta)
        self.base = ScriptedPusher(max_delta=self.max_delta)

    def reset(self, ref_xy):
        self.base.reset(ref_xy)

    def __call__(self, obs):
        base_action = self.base.act(obs)
        if self.base.done:
            return np.zeros(2)
        o = augment_obs(obs, base_action, self.max_delta)[None]
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(o)
        a, _ = self.model.predict(o, deterministic=True)
        cmd = executed_action(base_action, a[0], self.residual_scale, self.max_delta)
        self.base.executed(cmd)
        return cmd

    @property
    def done(self):
        return self.base.done
