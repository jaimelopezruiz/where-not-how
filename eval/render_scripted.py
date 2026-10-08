"""Sim-only GIF of the scripted pusher on one episode.

    python -m eval.render_scripted --split train --episode ep_025 --out results/scripted_train_ep_025.gif
"""
import argparse
import sys

import imageio
import numpy as np

from control.scripted_pusher import ScriptedPusher
from sim.push_env import PushTrackEnv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--episode", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=2, help="keep every n-th env step")
    args = ap.parse_args(argv)

    env = PushTrackEnv(split=args.split, render_mode="rgb_array")
    obs, _ = env.reset(options={"episode_id": args.episode})
    pusher = ScriptedPusher()
    pusher.reset(env.reference_path)
    frames, done, i = [env.render()], False, 0
    while not done:
        obs, _, terminated, truncated, _ = env.step(pusher(obs))
        done, i = terminated or truncated, i + 1
        if i % args.every == 0 or done:
            frames.append(env.render())
    env.close()
    imageio.mimsave(args.out, frames, duration=int(1000 * env.dt * args.every), loop=0)
    print(f"{args.episode}: {i} steps, {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
