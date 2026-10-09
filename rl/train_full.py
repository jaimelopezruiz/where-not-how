"""C6.4: PPO on the whole train split, ready to launch (nothing here touches the test split).

    python -m rl.train_full --name full_s0        # runs/full_s0, 8 envs, until stopped or 10 M steps (--name is required)
    python -m rl.train_full --seed 1 --name full_s1
    python -m rl.train_full --name full_s0 --resume   # continue from the latest checkpoint

Fixed seed, checkpoints (with VecNormalize statistics) every 500 k steps, TensorBoard and progress.csv in
runs/<name>/ (view with `tensorboard --logdir runs`). Every 100 k steps the deterministic policy is run on a
fixed subset of TRAIN episodes (the first two of each category, by id) under the until-done protocol and
logged to runs/<name>/eval_curve.csv. Evaluate the final model on the test split once, after training, with

    python -m rl.eval_policy --run runs/<name> --split test --out results/ppo_test
"""
import argparse

from eval.report import episode_categories
from rl.train_ppo import add_args, train


def eval_subset(split="train", per_category=2):
    """First `per_category` episodes of each category, by id, from the split (train only)."""
    import json

    from capture.common import REPO_ROOT
    ids = json.loads((REPO_ROOT / "data" / "splits.json").read_text())[split]
    cats, seen, out = episode_categories(), {}, []
    for ep in ids:
        c = cats[ep]
        if seen.get(c, 0) < per_category:
            seen[c] = seen.get(c, 0) + 1
            out.append(ep)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.set_defaults(name="full_s0")
    ap.add_argument("--total-steps", type=int, default=10_000_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    a = vars(ap.parse_args(argv))
    if a["split"] != "train":
        raise SystemExit("C6.4 trains and monitors on the train split only")
    train(episodes=None, eval_episodes=eval_subset("train"), **a)


if __name__ == "__main__":
    main()
