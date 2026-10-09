"""C6.5: residual PPO on the scripted pusher, ready to launch (nothing here touches the test split).

    python -m rl.train_residual --name residual_s0  # runs/residual_s0, 8 envs, 3 M steps (--name is required)
    python -m rl.train_residual --residual-scale 0.1 --name residual_s0_r01
    python -m rl.train_residual --name residual_s0 --resume

The policy outputs a residual in [-1, 1]^2 added to the scripted pusher's step (rl/residual.py), at most
residual_scale * max_delta. Same trainer, reward, train split, eval subset and PPO settings as rl.train_full
(C6.4), except log_std_init -2.0: small exploration, so training starts close to the scripted pusher.
Evaluate the final model on the test split once, after training, with

    python -m rl.eval_policy --run runs/<name> --split test --out results/residual_test
"""
import argparse

from rl.residual import RESIDUAL_SCALE
from rl.train_full import eval_subset
from rl.train_ppo import add_args, train


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.set_defaults(name="residual_s0", log_std_init=-2.0)
    ap.add_argument("--total-steps", type=int, default=3_000_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--checkpoint-every", type=int, default=500_000)
    ap.add_argument("--residual-scale", type=float, default=RESIDUAL_SCALE)
    ap.add_argument("--deviation-weight", type=float, default=None,
                    help="C6.7: displacement-weighted deviation penalty replacing -0.1 * lateral")
    a = vars(ap.parse_args(argv))
    if a["split"] != "train":
        raise SystemExit("C6.5 trains and monitors on the train split only")
    train(episodes=None, eval_episodes=eval_subset("train"), **a)


if __name__ == "__main__":
    main()
