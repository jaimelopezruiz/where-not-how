"""C4.2: results table, trajectory overlay plots, side-by-side GIF writer, and evaluate() stub.

evaluate() is the interface that connects a controller to the PushTrack-v0 gymnasium env
from T3. The env wiring is a TODO; everything else in this module is independent of the sim.
"""
import csv
from pathlib import Path
from typing import Callable

import imageio
import matplotlib
matplotlib.use("Agg")   # non-interactive; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

from eval.metrics import category_summary


# ── Results CSV ──────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "id", "category", "method",
    "progress", "mean_deviation_cm", "final_error_cm", "success", "completion_time",
]


def write_results_csv(results: list, path: "str | Path") -> None:
    """Write one row per episode to a CSV.

    Each element of results must have keys: id, category, method, and the four
    metric keys from eval.metrics.compute_metrics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ── Per-category summary table ────────────────────────────────────────────────

def category_table(results: list) -> str:
    """Return a plain-text per-category summary table.

    Parameters
    ----------
    results : list of episode dicts with 'category' and metric keys

    Returns
    -------
    Formatted string ready for printing or writing to a file.
    """
    summary = category_summary(results)
    header = (
        f"{'Category':<12} {'N':>4} {'Success%':>9} {'Mean dev (cm)':>14} "
        f"{'Final err (cm)':>15} {'Progress':>9} {'Time (s)':>9}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for cat, s in summary.items():
        t = s["mean_completion_time"]
        t_str = f"{t:.2f}" if not np.isnan(t) else "n/a"
        lines.append(
            f"{cat:<12} {s['n']:>4} {s['success_rate'] * 100:>9.1f} "
            f"{s['mean_deviation_cm']:>14.2f} "
            f"{s['mean_final_error_cm']:>15.2f} "
            f"{s['mean_progress']:>9.3f} "
            f"{t_str:>9}"
        )
    return "\n".join(lines)


# ── Overlay plot ──────────────────────────────────────────────────────────────

def plot_overlay(
    episodes: list,
    path: "str | Path",
    title: str = "Reference vs achieved trajectories",
) -> None:
    """Plot reference and achieved paths for a set of episodes.

    Parameters
    ----------
    episodes : list of dicts, each with:
                 ref_xy      (R, 2) metres
                 achieved_xy (N, 2) metres
               Optional: 'id', 'category', 'success'
    path     : output image path (.png or .pdf)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    cat_seen: set = set()

    for ep in episodes:
        ref_xy = np.asarray(ep["ref_xy"])
        ach_xy = np.asarray(ep["achieved_xy"])
        cat = ep.get("category", "")
        success = ep.get("success", None)

        label_ref = None
        if cat not in cat_seen:
            label_ref = f"ref ({cat})" if cat else "reference"
            cat_seen.add(cat)

        ax.plot(
            ref_xy[:, 0] * 100, ref_xy[:, 1] * 100,
            "--", color="grey", alpha=0.5, linewidth=1.0, label=label_ref,
        )
        color = "tab:green" if success else ("tab:red" if success is False else "tab:blue")
        ax.plot(
            ach_xy[:, 0] * 100, ach_xy[:, 1] * 100,
            "-", color=color, alpha=0.65, linewidth=1.0,
        )

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    if cat_seen:
        ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ── Side-by-side GIF ──────────────────────────────────────────────────────────

def write_gif(
    real_frames: list,
    sim_frames: list,
    path: "str | Path",
    fps: float = 10.0,
) -> None:
    """Write a side-by-side GIF from real-video and sim-render frame sequences.

    Parameters
    ----------
    real_frames : list of (H, W, 3) or (H, W) uint8 arrays  (real video)
    sim_frames  : list of (H', W', 3) or (H', W') uint8 arrays  (sim render)
    path        : output .gif path
    fps         : frames per second

    The two sequences must have the same length. Frames are padded to the same
    height and concatenated horizontally.
    """
    if len(real_frames) != len(sim_frames):
        raise ValueError(
            f"Frame count mismatch: {len(real_frames)} real vs {len(sim_frames)} sim"
        )
    if len(real_frames) == 0:
        raise ValueError("Frame sequences are empty")

    def _normalise(frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 2:                              # greyscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:                        # RGBA -> RGB
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    combined = []
    for real, sim in zip(real_frames, sim_frames):
        r = _normalise(real)
        s = _normalise(sim)
        hr, wr = r.shape[:2]
        hs, ws = s.shape[:2]
        h = max(hr, hs)
        if hr < h:
            r = np.pad(r, ((0, h - hr), (0, 0), (0, 0)))
        if hs < h:
            s = np.pad(s, ((0, h - hs), (0, 0), (0, 0)))
        combined.append(np.concatenate([r, s], axis=1))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), combined, duration=int(1000 / fps), loop=0)


# ── Evaluate interface ────────────────────────────────────────────────────────

def evaluate(controller: Callable, split: str) -> list:
    """Run controller on every episode in split and return per-episode metrics.

    Parameters
    ----------
    controller : callable  obs -> action (planar EE velocity, as in C3.3)
    split      : one of 'train', 'test', 'showcase'

    Returns
    -------
    list of dicts, one per episode, with keys:
      id, category, method, progress, mean_deviation_cm, final_error_cm,
      success, completion_time

    TODO: wire to PushTrack-v0 gymnasium env from T3 (sim/env.py) when that
    branch merges. Steps:
      1. Load data/splits.json to get episode ids for the requested split.
      2. For each episode id:
         a. gym.make("PushTrack-v0", episode=ep_id) and reset().
         b. Roll out: obs, done = env.reset(), False
            while not done: obs, _, done, _, info = env.step(controller(obs))
         c. Retrieve ref_xy and achieved_xy from info (or env attributes).
         d. Call compute_metrics(ref_xy, achieved_xy, t) and append to results.
      3. Return results list.
    """
    raise NotImplementedError(
        "evaluate() requires PushTrack-v0 from T3 (sim/env.py). "
        "Connect the env here when T3 merges into main."
    )
