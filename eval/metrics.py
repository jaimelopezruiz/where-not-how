"""C4.1: evaluation metrics for cube-pushing trajectories.

All position inputs are in metres; outputs are in centimetres where noted.
All functions are pure numpy with no I/O side-effects.
"""
import numpy as np

SUCCESS_FINAL_M = 0.02   # final error must be strictly less than 2 cm
SUCCESS_PROGRESS = 0.9   # progress must be >= 90 %


def _polyline_project(ref_xy: np.ndarray, pts: np.ndarray):
    """Project each point in pts onto the nearest point on the polyline ref_xy.

    Parameters
    ----------
    ref_xy : (R, 2)  reference polyline vertices, metres
    pts    : (N, 2)  query points, metres

    Returns
    -------
    arc_coords   : (N,)  arc-length coordinate of the nearest polyline point, metres
    distances    : (N,)  Euclidean distance to the polyline, metres
    total_length : float  total polyline arc length, metres
    """
    if len(ref_xy) < 2:
        raise ValueError("ref_xy must have at least 2 vertices")

    segs = np.diff(ref_xy, axis=0)                          # (R-1, 2)
    seg_lens = np.hypot(segs[:, 0], segs[:, 1])            # (R-1,)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])  # (R,)
    total_len = float(cum_len[-1])

    # Projection parameter t in [0, 1] for each (point, segment) pair.
    # Shape: pts (N, 1, 2) - ref[:-1] (1, R-1, 2) -> AP (N, R-1, 2)
    AP = pts[:, None, :] - ref_xy[None, :-1, :]
    denom = np.maximum(seg_lens ** 2, 1e-24)               # avoid /0 on zero-length seg
    t = np.sum(AP * segs[None, :, :], axis=2) / denom      # (N, R-1)
    t = np.clip(t, 0.0, 1.0)

    nearest = ref_xy[None, :-1, :] + t[:, :, None] * segs[None, :, :]  # (N, R-1, 2)
    dist_sq = np.sum((pts[:, None, :] - nearest) ** 2, axis=2)          # (N, R-1)

    best_seg = np.argmin(dist_sq, axis=1)                  # (N,)
    idx = np.arange(len(pts))
    arc_coords = cum_len[best_seg] + t[idx, best_seg] * seg_lens[best_seg]
    distances = np.sqrt(dist_sq[idx, best_seg])

    return arc_coords, distances, total_len


def compute_metrics(
    ref_xy: np.ndarray,
    achieved_xy: np.ndarray,
    t: "np.ndarray | None" = None,
) -> dict:
    """Episode-level metrics for one cube-pushing run.

    Parameters
    ----------
    ref_xy      : (R, 2)  reference cube trajectory, metres
    achieved_xy : (N, 2)  achieved cube trajectory, metres
    t           : (N,)    sample timestamps, seconds (required for completion_time)

    Returns
    -------
    dict with keys:
      progress          float [0..1]  arc-length fraction of the reference reached
      mean_deviation_cm float         mean distance to the reference polyline, cm
      final_error_cm    float         distance from last achieved point to ref end, cm
      success           bool          final_error < 2 cm AND progress >= 0.9
      completion_time   float or nan  time of first step where success holds; nan if none
    """
    ref_xy = np.asarray(ref_xy, dtype=float)
    achieved_xy = np.asarray(achieved_xy, dtype=float)

    arc_coords, distances, total_len = _polyline_project(ref_xy, achieved_xy)

    # Progress: furthest arc-length position reached, normalised.
    if total_len > 0:
        progress = float(min(np.max(arc_coords) / total_len, 1.0))
    else:
        progress = 0.0

    # Mean deviation: arc-length-weighted average of pointwise distances.
    # Each achieved point i>0 is weighted by its step size ds_i so that
    # time spent stationary (ds_i = 0) contributes nothing to the score.
    # Fallback to the start-point deviation when the cube never moves.
    diffs = np.diff(achieved_xy, axis=0)                            # (N-1, 2)
    ds = np.hypot(diffs[:, 0], diffs[:, 1])                        # (N-1,) step sizes
    total_ds = float(np.sum(ds))
    if total_ds > 0:
        mean_deviation_cm = float(np.sum(distances[1:] * ds) / total_ds * 100.0)
    else:
        mean_deviation_cm = float(distances[0] * 100.0)

    # Final error: distance from the last achieved point to the reference endpoint (cm).
    final_error_cm = float(np.linalg.norm(achieved_xy[-1] - ref_xy[-1]) * 100.0)

    # Episode-level success.
    success = (final_error_cm < SUCCESS_FINAL_M * 100.0) and (progress >= SUCCESS_PROGRESS)

    # Completion time: first timestep where success holds simultaneously.
    completion_time = float("nan")
    if t is not None:
        t_arr = np.asarray(t, dtype=float)
        # Monotone progress up to each step.
        if total_len > 0:
            running_progress = np.minimum(
                np.maximum.accumulate(arc_coords) / total_len, 1.0
            )
        else:
            running_progress = np.zeros(len(achieved_xy))
        step_final_err = np.linalg.norm(achieved_xy - ref_xy[-1], axis=1)
        success_mask = (step_final_err < SUCCESS_FINAL_M) & (running_progress >= SUCCESS_PROGRESS)
        first = int(np.argmax(success_mask))
        if success_mask[first]:
            completion_time = float(t_arr[first])

    return {
        "progress": progress,
        "mean_deviation_cm": mean_deviation_cm,
        "final_error_cm": final_error_cm,
        "success": success,
        "completion_time": completion_time,
    }


def _nanmean(vals: list) -> float:
    finite = [float(v) for v in vals if not np.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def category_summary(results: list) -> dict:
    """Per-category aggregates of episode metrics.

    Parameters
    ----------
    results : list of dicts, each with 'category' and the metric keys from compute_metrics

    Returns
    -------
    dict mapping category -> {n, success_rate, mean_progress, mean_deviation_cm,
                               mean_final_error_cm, mean_completion_time}
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r)

    summary = {}
    for cat, rows in sorted(groups.items()):
        summary[cat] = {
            "n": len(rows),
            "success_rate": float(np.mean([float(r["success"]) for r in rows])),
            "mean_progress": float(np.mean([r["progress"] for r in rows])),
            "mean_deviation_cm": float(np.mean([r["mean_deviation_cm"] for r in rows])),
            "mean_final_error_cm": float(np.mean([r["final_error_cm"] for r in rows])),
            "mean_completion_time": _nanmean([r["completion_time"] for r in rows]),
        }
    return summary
