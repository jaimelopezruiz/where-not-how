"""Evaluate a trained PPO policy under the until-done protocol (eval.metrics, same as the scripted pusher).

    python -m rl.eval_policy --run runs/overfit_ep010_a1 --split train --episodes ep_010 --out results/ppo_overfit_ep010

--model picks a checkpoint instead of the final model.zip (its VecNormalize statistics are found next to it).
"""
import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eval.report import category_table, evaluate, plot_overlay, write_results_csv
from rl.policy_controller import PolicyController
from rl.train_ppo import make_env


def load_controller(run, model=None, k_ref=5, residual_scale=None):
    """PolicyController from runs/<name>: final model.zip + vecnormalize.pkl, or a checkpoint zip.
    A residual run (residual_scale in its config.json) gives a ResidualController instead."""
    run = Path(run)
    if model is None:
        zip_path, stats = run / "model.zip", run / "vecnormalize.pkl"
    else:
        zip_path = Path(model)
        stats = zip_path.with_name(zip_path.name.replace("rl_model_", "rl_model_vecnormalize_").replace(".zip", ".pkl"))
    dummy = DummyVecEnv([lambda: make_env("train", None, 300, k_ref, 0, residual_scale)])
    vecnorm = VecNormalize.load(str(stats), dummy)
    vecnorm.training = False
    policy = PPO.load(str(zip_path), device="cpu")
    if residual_scale is not None:
        from rl.residual import ResidualController
        return ResidualController(policy, vecnorm, residual_scale)
    return PolicyController(policy, vecnorm, k_ref=k_ref)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model")
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--max-episode-steps", type=int)
    ap.add_argument("--method", default="ppo")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = json.loads((Path(a.run) / "config.json").read_text())
    k_ref = cfg.get("k_ref", 5)
    steps = a.max_episode_steps or cfg.get("max_episode_steps", 600)
    ctrl = load_controller(a.run, a.model, k_ref, cfg.get("residual_scale"))
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(split=a.split, episodes=a.episodes, max_episode_steps=steps, k_ref=k_ref)
    ctrl.max_delta = env.max_delta
    results = evaluate(ctrl, env=env, method=a.method)
    env.close()
    text = category_table(results)
    ok = sum(bool(r["success"]) for r in results)
    print(text, f"\n{ok}/{len(results)} succeed")
    for r in results:
        print(f"{r['id']}: success={r['success']} progress={r['progress']:.2f} dev={r['mean_deviation_cm']:.2f} cm "
              f"final={r['final_error_cm']:.2f} cm steps={r['steps']}")
    if a.out:
        out = Path(a.out)
        write_results_csv(results, out.parent / (out.name + ".csv"))
        plot_overlay(results, out.parent / (out.name + "_overlay.png"), title=f"{a.method}, {a.split}")
        (out.parent / (out.name + "_summary.txt")).write_text(text + f"\n{ok}/{len(results)} succeed\n")
    return results


if __name__ == "__main__":
    main()
