"""PPO on PushTrack-v0 (C6.3 overfit one trajectory; C6.4 train split uses the same trainer).

    python -m rl.train_ppo --name overfit_ep010 --episodes ep_010 --eval-episodes ep_010 --max-episode-steps 300 --time-limit-min 20

Outputs under runs/<name>/ (gitignored): TensorBoard logs and progress.csv (SB3 logger), eval_curve.csv
(deterministic policy on the eval episodes under the until-done protocol), checkpoints with their
VecNormalize statistics, final model.zip and vecnormalize.pkl.

Actions are rescaled to [-1, 1] for the policy (the env's own action space is +-max_delta metres).
"""
import argparse
import csv
import json
import os
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rl.policy_controller import PolicyController

RUNS = Path(__file__).resolve().parent.parent / "runs"


class PolicyActions(gym.Wrapper):
    """Policy-facing env: actions in [-1, 1] scaled to metres; success flag exposed as info['is_success']."""

    def __init__(self, env):
        super().__init__(env)
        self.max_delta = float(env.unwrapped.max_delta)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        a = np.clip(action, -1.0, 1.0) * self.max_delta
        obs, reward, terminated, truncated, info = self.env.step(a)
        info["is_success"] = bool(info.get("success", False))
        return obs, reward, terminated, truncated, info


def make_env(split, episodes, max_episode_steps, k_ref, seed):
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(split=split, episodes=episodes, max_episode_steps=max_episode_steps, k_ref=k_ref)
    env = Monitor(PolicyActions(env), info_keywords=("is_success",))
    env.reset(seed=seed)
    return env


class WallClockLimit(BaseCallback):
    def __init__(self, minutes):
        super().__init__()
        self.minutes, self.deadline = minutes, None

    def _on_training_start(self):
        self.deadline = None if self.minutes is None else time.time() + 60 * self.minutes

    def _on_step(self):
        return self.deadline is None or time.time() < self.deadline


class EvalCurve(BaseCallback):
    """Deterministic policy on fixed episodes, scored by eval.metrics under the until-done protocol."""

    def __init__(self, split, episodes, every, run_dir, max_episode_steps, k_ref, label="eval"):
        super().__init__()
        self.split, self.episodes, self.every = split, episodes, every
        self.path = Path(run_dir) / f"{label}_curve.csv"
        self.label, self.max_episode_steps, self.k_ref = label, max_episode_steps, k_ref
        self._next, self._env, self._t0 = 0, None, time.time()

    def _on_training_start(self):
        from sim.push_env import PushTrackEnv
        self._env = PushTrackEnv(split=self.split, episodes=self.episodes,
                                 max_episode_steps=self.max_episode_steps, k_ref=self.k_ref)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(["timesteps", "wall_s", "success_rate", "mean_deviation_cm",
                                        "final_error_cm", "progress", "steps"])

    def _evaluate(self):
        from eval.report import evaluate
        ctrl = PolicyController(self.model, self.model.get_vec_normalize_env(), k_ref=self.k_ref,
                                max_delta=self._env.max_delta)
        res = evaluate(ctrl, env=self._env, method="ppo")
        row = [self.num_timesteps, round(time.time() - self._t0), np.mean([r["success"] for r in res]),
               np.mean([r["mean_deviation_cm"] for r in res]), np.mean([r["final_error_cm"] for r in res]),
               np.mean([r["progress"] for r in res]), np.mean([r["steps"] for r in res])]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        for key, v in zip(("success_rate", "mean_deviation_cm", "final_error_cm", "progress"), row[2:6]):
            self.logger.record(f"{self.label}/{key}", float(v))
        return row

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next = self.num_timesteps + self.every
            self._evaluate()
        return True

    def _on_training_end(self):
        self._evaluate()
        self._env.close()


def train(name, split="train", episodes=None, eval_episodes=None, n_envs=8, seed=0, total_steps=50_000_000,
          time_limit_min=None, max_episode_steps=600, k_ref=5, eval_every=50_000, checkpoint_every=250_000,
          n_steps=512, batch_size=256, lr=3e-4, ent_coef=0.0, gamma=0.99, log_std_init=-1.0, resume=False,
          torch_threads=2):
    os.environ.setdefault("OMP_NUM_THREADS", "1")      # env workers inherit this: one BLAS thread each
    torch.set_num_threads(torch_threads)
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    cfg = dict(locals())
    cfg.pop("run")
    (run / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    fns = [partial(make_env, split, episodes, max_episode_steps, k_ref, seed + i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn")
    ckpt = sorted(run.glob("rl_model_*_steps.zip"), key=lambda p: int(p.stem.split("_")[2]))
    resuming = bool(resume and ckpt)
    if resuming:
        stats = ckpt[-1].with_name(ckpt[-1].name.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"))
        venv = VecNormalize.load(str(stats), venv)
        model = PPO.load(str(ckpt[-1]), env=venv, device="cpu")
        print(f"resumed from {ckpt[-1].name} at {model.num_timesteps} steps")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=gamma)
        model = PPO("MlpPolicy", venv, seed=seed, n_steps=n_steps, batch_size=batch_size, n_epochs=10, gamma=gamma,
                    gae_lambda=0.95, learning_rate=lr, ent_coef=ent_coef, clip_range=0.2, device="cpu",
                    policy_kwargs={"log_std_init": log_std_init}, verbose=0)
    model.set_logger(configure(str(run), ["stdout", "csv", "tensorboard"]))
    callbacks = [
        CheckpointCallback(max(checkpoint_every // n_envs, 1), str(run), name_prefix="rl_model", save_vecnormalize=True),
        EvalCurve(split, eval_episodes or episodes, eval_every, run, max_episode_steps, k_ref),
        WallClockLimit(time_limit_min),
    ]
    t0 = time.time()
    model.learn(total_steps, callback=callbacks, reset_num_timesteps=not resuming)
    model.save(str(run / "model"))
    venv.save(str(run / "vecnormalize.pkl"))
    venv.close()
    print(f"done: {model.num_timesteps} steps in {time.time() - t0:.0f} s -> {run}")
    return run


def add_args(ap):
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-limit-min", type=float)
    ap.add_argument("--max-episode-steps", type=int, default=600)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--log-std-init", type=float, default=-1.0)
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in runs/<name>")
    ap.add_argument("--torch-threads", type=int, default=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--episodes", nargs="*", help="restrict the training env to these episode ids")
    ap.add_argument("--eval-episodes", nargs="*")
    ap.add_argument("--total-steps", type=int, default=50_000_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--checkpoint-every", type=int, default=250_000)
    train(**vars(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
