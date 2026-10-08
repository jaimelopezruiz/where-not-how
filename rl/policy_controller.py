"""A trained PPO policy as an evaluation controller (obs -> action, with a stopping rule).

eval.report.rollout runs a controller until it reports done. A policy has no done output, so this wrapper
supplies one from the observation alone: once every observed reference point has clamped to the path end
and the cube is within `arrive_tol` of it, the policy is handed over to a hold (zero action) for
`hold_steps` steps, so the cube can settle, then the episode ends. This is the stopping rule the scripted
pusher has (stop_tol), and it lets the protocol measure where the box really ends.
"""
import numpy as np


class PolicyController:
    def __init__(self, model, vec_normalize=None, k_ref=5, arrive_tol=0.02, hold_steps=20, max_delta=0.02):
        self.model, self.vecnorm = model, vec_normalize
        self.k_ref, self.arrive_tol, self.hold_steps, self.max_delta = k_ref, arrive_tol, hold_steps, max_delta
        self.reset(None)

    def reset(self, ref_xy):
        self._hold = None            # None until arrived, then steps held so far

    def _arrived(self, obs):
        ref = np.asarray(obs[6:6 + 2 * self.k_ref], float).reshape(-1, 2)
        clamped = np.ptp(ref, axis=0).max() < 1e-4        # every point is the path end
        return clamped and np.hypot(*ref[0]) < self.arrive_tol

    def __call__(self, obs):
        if self._hold is None and self._arrived(obs):
            self._hold = 0
        if self._hold is not None:
            self._hold += 1
            return np.zeros(2)
        o = np.asarray(obs, np.float32)[None]
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(o)
        action, _ = self.model.predict(o, deterministic=True)
        return np.clip(action[0], -1.0, 1.0) * self.max_delta     # the policy acts in [-1, 1]

    @property
    def done(self):
        return self._hold is not None and self._hold > self.hold_steps
