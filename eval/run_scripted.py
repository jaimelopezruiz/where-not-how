"""C5.2: evaluate the scripted pusher on a split.

    python -m eval.run_scripted --split train --push-speed 0.005 --note "slower push"
    python -m eval.run_scripted --split test --out results/scripted_test   # final run, once

Every run appends one line to results/scripted_tuning_log.csv (parameters and per-category success).
With --out PREFIX it also writes PREFIX.csv, PREFIX_overlay.png and PREFIX_summary.txt.
"""
import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

from control.scripted_pusher import ScriptedPusher
from eval.metrics import category_summary
from eval.report import category_table, evaluate, plot_overlay, write_results_csv

LOG = Path(__file__).resolve().parent.parent / "results" / "scripted_tuning_log.csv"
PARAMS = ("max_delta", "lookahead", "pusher_radius", "margin", "push_speed", "stop_tol", "lat_tol",
          "lat_exit", "progress_window", "k_lat")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "test", "showcase"))
    ap.add_argument("--method", default="scripted")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", help="prefix for the CSV, overlay and summary files")
    ap.add_argument("--note", default="")
    ap.add_argument("--first-success", action="store_true",
                    help="stop each episode at the env's first success (training behaviour) instead of when "
                         "the pusher reports done")
    sig = inspect.signature(ScriptedPusher.__init__).parameters
    for name in PARAMS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=float, default=sig[name].default)
    args = ap.parse_args(argv)
    params = {k: getattr(args, k) for k in PARAMS}
    controller = ScriptedPusher(**params)

    t0 = time.time()
    results = evaluate(controller, args.split, method=args.method, workers=args.workers,
                       until_done=not args.first_success)
    elapsed = time.time() - t0

    table = category_table(results)
    n = len(results)
    overall = {
        "success": float(np.mean([r["success"] for r in results])),
        "dev": float(np.mean([r["mean_deviation_cm"] for r in results])),
        "final": float(np.mean([r["final_error_cm"] for r in results])),
        "progress": float(np.mean([r["progress"] for r in results])),
    }
    text = (f"{args.method} on {args.split} ({n} episodes, {elapsed:.0f} s)\nparams: {json.dumps(params)}\n\n{table}\n\n"
            f"overall: success {overall['success'] * 100:.1f}%  mean deviation {overall['dev']:.2f} cm  "
            f"final error {overall['final']:.2f} cm  progress {overall['progress']:.3f}\n")
    print(text)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "split", "note", *PARAMS, "n_ok", "n", "success", "mean_dev_cm", "final_err_cm", "progress",
                        "success_by_category"])
        by_cat = {c: round(s["success_rate"], 2) for c, s in category_summary(results).items()}
        w.writerow([time.strftime("%Y-%m-%d %H:%M"), args.split, args.note, *[params.get(k, "") for k in PARAMS],
                    sum(bool(r["success"]) for r in results), n, round(overall["success"], 3), round(overall["dev"], 2),
                    round(overall["final"], 2), round(overall["progress"], 3), json.dumps(by_cat)])

    if args.out:
        out = Path(args.out)
        write_results_csv(results, out.parent / (out.name + ".csv"))
        plot_overlay(results, out.parent / (out.name + "_overlay.png"),
                     title=f"{args.method}, {args.split}: reference (dashed) vs achieved cube path")
        (out.parent / (out.name + "_summary.txt")).write_text(text)
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
