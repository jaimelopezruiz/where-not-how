"""C11.3: results table and training-curve figure.

    python -m scripts.results_figures results/scripted_test.csv results/residual_test.csv
    python -m scripts.results_figures --run "plain PPO=runs/full_s0" --run "residual=runs/residual_s0" \\
        results/scripted_test.csv

Table: each CSV argument is a results/<method>_test.csv from eval.report.write_results_csv. Printed and written
to results/results_table.md as markdown: one overall table (success, mean deviation, final error, completion
time) and one by category. Rows follow the argument order, the method name is the CSV's `method` column. The
numbers are aggregated with eval.metrics.category_summary, as the per-run summaries are. Completion time is the
mean over the successful episodes (failed episodes have none).

Curves: each --run NAME=DIR gives one line per panel, against timesteps.
  deterministic eval success and mean deviation: DIR/eval_curve*.csv (the eval episodes, EvalCurve in
  rl/train_ppo.py); rollout success: DIR/progress*.csv (the SB3 logger, column rollout/success_rate; a resumed
  run has progress_0-3M.csv beside progress.csv because the logger rewrites progress.csv on resume).
  Several files of one kind are concatenated and sorted by timesteps; a timestep that appears twice keeps the
  later file's row. --progress NAME=FILE / --eval NAME=FILE (repeatable) name the files explicitly instead.
Written to results/training_curves.png.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from capture.common import REPO_ROOT
from eval.metrics import category_summary

# categorical slots 1 and 2 of the reference palette, light surface
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


# -- table --------------------------------------------------------------------------------------------------

def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read_results(path):
    """(method, rows) of a results CSV; rows have the typed fields eval.metrics.category_summary needs."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "category": r["category"], "method": r.get("method", ""),
                         "success": r["success"].strip().lower() == "true", "progress": _float(r["progress"]),
                         "mean_deviation_cm": _float(r["mean_deviation_cm"]),
                         "final_error_cm": _float(r["final_error_cm"]), "completion_time": _float(r["completion_time"])})
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return (rows[0]["method"] or Path(path).stem.removesuffix("_test")), rows


def _cells(s):
    t = s["mean_completion_time"]
    return (f"{round(s['success_rate'] * s['n'])}/{s['n']} ({s['success_rate'] * 100:.0f}%)",
            f"{s['mean_deviation_cm']:.2f}", f"{s['mean_final_error_cm']:.2f}", "n/a" if math.isnan(t) else f"{t:.1f}")


def results_table(files):
    """Markdown with an overall table and a by-category table for the results CSVs, in the order given."""
    loaded = [read_results(p) for p in files]
    head = "| Method | Episodes | Success | Mean deviation (cm) | Final error (cm) | Completion time (s) |"
    overall = [head, "|---|--:|--:|--:|--:|--:|"]
    for method, rows in loaded:
        s = category_summary([{**r, "category": "all"} for r in rows])["all"]
        overall.append(f"| {method} | {s['n']} | " + " | ".join(_cells(s)) + " |")
    by_cat = ["| Method | Category | Episodes | Success | Mean deviation (cm) | Final error (cm) | Completion time (s) |",
              "|---|---|--:|--:|--:|--:|--:|"]
    for method, rows in loaded:
        for cat, s in category_summary(rows).items():
            by_cat.append(f"| {method} | {cat} | {s['n']} | " + " | ".join(_cells(s)) + " |")
    src = ", ".join(f"`{Path(p).as_posix()}`" for p in files)
    return ("**Overall**\n\n" + "\n".join(overall) + "\n\n**By category**\n\n" + "\n".join(by_cat) +
            f"\n\nCompletion time is the mean over successful episodes. Source: {src}.\n")


# -- curves -------------------------------------------------------------------------------------------------

def read_curve(paths, step_col, value_cols):
    """Concatenate CSVs of one run into {col: array} sorted by timesteps, under key "t". Rows without a finite
    step or value are dropped per column; a repeated timestep keeps the later file's row."""
    by_col = {c: {} for c in value_cols}
    for path in paths:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                t = _float(r.get(step_col))
                if not math.isfinite(t):
                    continue
                for c in value_cols:
                    v = _float(r.get(c))
                    if math.isfinite(v):
                        by_col[c][t] = v
    out = {}
    for c, d in by_col.items():
        ts = sorted(d)
        out[c] = (np.array(ts), np.array([d[t] for t in ts]))
    return out


def moving_average(y, window):
    if len(y) < 2 or window < 2:
        return y
    k = min(window, len(y))
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def _first_step(path):
    """Timestep of the first row of a curve CSV (its first column holds timesteps in eval curves; progress files
    carry them in time/total_timesteps)."""
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for key in ("timesteps", "time/total_timesteps"):
                t = _float(r.get(key))
                if math.isfinite(t):
                    return t
    return math.inf


def run_files(spec, explicit, pattern):
    """Files of one kind for one run: the explicit ones if given, else DIR/<pattern> in name order."""
    if explicit:
        return explicit
    d = Path(spec)
    files = sorted(d.glob(pattern), key=_first_step)      # a resumed run's later file starts at a later timestep
    if not files:
        raise SystemExit(f"{d}: no {pattern} (pass --{'eval' if 'eval' in pattern else 'progress'} NAME=FILE)")
    return files


def plot_curves(runs, path, smooth=25):
    """runs: list of (name, eval_files, progress_files). Three panels against timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), facecolor=SURFACE)
    titles = ["Deterministic eval: success rate", "Deterministic eval: mean deviation (cm)",
              "Training rollouts: success rate"]
    for i, (name, eval_files, progress_files) in enumerate(runs):
        colour, marker = COLOURS[i % len(COLOURS)], MARKERS[i % len(MARKERS)]
        ev = read_curve(eval_files, "timesteps", ["success_rate", "mean_deviation_cm"])
        pr = read_curve(progress_files, "time/total_timesteps", ["rollout/success_rate"])
        for ax, (t, y) in zip(axes[:2], (ev["success_rate"], ev["mean_deviation_cm"])):
            ax.plot(t / 1e6, y, "-", color=colour, lw=1.8, marker=marker, ms=4.5, mec=SURFACE, mew=1.0,
                    label=name)
        t, y = pr["rollout/success_rate"]
        axes[2].plot(t / 1e6, y, "-", color=colour, lw=0.8, alpha=0.3)
        axes[2].plot(t / 1e6, moving_average(y, smooth), "-", color=colour, lw=2.0, label=name)
    for ax, title in zip(axes, titles):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("timesteps (millions)", fontsize=9, color=INK_2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xlim(left=0)
    axes[0].set_ylim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[1].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.012, f"Deviation is measured over the box's displacement, so a box that never moves reads near 0 "
             f"(check it against success). Rollout panel: thin line per iteration, bold {smooth}-iteration mean.",
             fontsize=8, color=INK_2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, facecolor=SURFACE)
    plt.close(fig)


# -- main ---------------------------------------------------------------------------------------------------

def _named(items, what):
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"{what} expects NAME=PATH, got {item!r}")
        out.setdefault(name, []).append(Path(value))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="*", help="results/<method>_test.csv files for the table")
    ap.add_argument("--run", action="append", help="NAME=runs/<dir>: a training run for the curves (repeatable)")
    ap.add_argument("--progress", action="append", help="NAME=FILE: progress CSV of run NAME instead of DIR/progress*.csv")
    ap.add_argument("--eval", action="append", help="NAME=FILE: eval curve CSV of run NAME instead of DIR/eval_curve*.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--smooth", type=int, default=25, help="moving-average window of the rollout curve (iterations)")
    args = ap.parse_args(argv)
    if not args.results and not args.run:
        ap.error("give results CSVs for the table and/or --run for the curves")

    if args.results:
        text = results_table(args.results)
        print(text)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "results_table.md").write_text(text, newline="\n")
    if args.run:
        explicit_p, explicit_e = _named(args.progress, "--progress"), _named(args.eval, "--eval")
        runs = []
        for spec in args.run:
            name, sep, d = spec.partition("=")
            if not sep:
                raise SystemExit(f"--run expects NAME=DIR, got {spec!r}")
            runs.append((name, run_files(d, explicit_e.get(name), "eval_curve*.csv"),
                         run_files(d, explicit_p.get(name), "progress*.csv")))
        plot_curves(runs, args.out_dir / "training_curves.png", args.smooth)
        print(f"{args.out_dir / 'training_curves.png'}: " + ", ".join(
            f"{n} ({len(e)} eval, {len(p)} progress files)" for n, e, p in runs))


if __name__ == "__main__":
    main(sys.argv[1:])
