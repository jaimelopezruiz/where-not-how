"""Cleaning (C2.4): confidence gate, short-gap interpolation, Savitzky-Golay smoothing, resampling, contact.

Operates on (x, y, yaw) series on the camera's own timestamps. Order matters:
  1. frames the pose solver or the geometry make doubtful are dropped (NaN);
  2. yaw is unwrapped, then gaps of at most MAX_GAP frames are linearly interpolated; longer gaps stay NaN;
  3. each unbroken stretch is smoothed with a Savitzky-Golay filter on the true timestamps (never across a gap);
  4. the result is resampled to the control rate on the true timestamps; grid points inside a gap stay NaN.
"""
from dataclasses import dataclass

import numpy as np

CONTROL_HZ = 20.0         # output rate; the sim env may resample further, t is stored with every episode
MAX_GAP = 5               # frames; longer gaps are not bridged
CUBE_RMS_MAX = 1.5        # px reprojection error of the cube marker (clean frames sit at 0.1-0.8)
TILT_MAX = np.radians(20)  # marker normal vs table normal; a box lying flat reads 0-10 deg, IPPE flips read far more
HAMPEL_WINDOW = 7         # frames
HAMPEL_XY = 0.010         # m from the local median
HAMPEL_YAW = np.radians(15)
SG_WINDOW = 11            # frames (0.37 s at 30 fps)
SG_ORDER = 2
CONTACT_SPEED = 0.005     # m/s; the box counts as pushed above this speed
CONTACT_MIN_RUN_S = 0.2
CONTACT_FILL_S = 0.5      # pauses shorter than this inside a push stay in contact


def unwrap_nan(angle):
    """np.unwrap over the finite samples only, so a NaN gap does not break the branch."""
    out = angle.copy()
    ok = np.isfinite(out)
    out[ok] = np.unwrap(out[ok])
    return out


def hampel_outliers(values, window=HAMPEL_WINDOW, xy_tol=HAMPEL_XY, yaw_tol=HAMPEL_YAW):
    """Frames whose position or (unwrapped) yaw is far from the median of its neighbours.

    values: (N, 3) x, y, yaw with NaN allowed. Catches single-frame pose flips the reprojection error misses.
    """
    n = len(values)
    half = window // 2
    bad = np.zeros(n, bool)
    for i in range(n):
        if not np.isfinite(values[i, 0]):
            continue
        nb = values[max(0, i - half):i + half + 1]
        nb = nb[np.isfinite(nb[:, 0])]
        if len(nb) < 3:
            continue
        med = np.median(nb, axis=0)
        bad[i] = (np.hypot(*(values[i, :2] - med[:2])) > xy_tol) or (abs(values[i, 2] - med[2]) > yaw_tol)
    return bad


def confidence_mask(detected, cube_rms, tilt, values):
    """True for frames kept: detected, low reprojection error, near-flat marker, not a local outlier."""
    ok = detected & np.isfinite(values[:, 0]) & (cube_rms <= CUBE_RMS_MAX) & (tilt <= TILT_MAX)
    v = values.copy()
    v[~ok] = np.nan
    v[:, 2] = unwrap_nan(v[:, 2])
    return ok & ~hampel_outliers(v)


def interpolate_gaps(t, values, valid, max_gap=MAX_GAP):
    """Linearly fill runs of at most ``max_gap`` invalid frames that lie between valid ones.

    values: (N, C), already on one continuous branch (unwrap yaw first). Returns (values, ok) where ok marks
    valid or filled frames; everything else is NaN. Runs at the start or end of the series are never filled.
    """
    out = np.where(valid[:, None], values, np.nan)
    ok = valid.copy()
    n, i = len(valid), 0
    while i < n:
        if valid[i]:
            i += 1
            continue
        j = i
        while j < n and not valid[j]:
            j += 1
        if i > 0 and j < n and j - i <= max_gap:
            for c in range(values.shape[1]):
                out[i:j, c] = np.interp(t[i:j], [t[i - 1], t[j]], [values[i - 1, c], values[j, c]])
            ok[i:j] = True
        i = j
    return out, ok


def segments(ok):
    """[start, stop) index pairs of the runs of True."""
    edges = np.diff(np.concatenate([[0], ok.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def savgol_times(t, values, window, order):
    """Savitzky-Golay smoothing on non-uniform sample times: a least-squares polynomial of ``order`` over the
    ``window`` nearest samples (shifted inward at the ends, like scipy's mode="interp"), evaluated at each sample.

    On uniform times this is scipy.signal.savgol_filter; on the camera's jittery timestamps it still treats
    every sample at the time it was stamped, which a filter over frame indices would not.
    """
    n = len(t)
    out = np.empty_like(values)
    for i in range(n):
        lo = min(max(i - window // 2, 0), n - window)
        sl = slice(lo, lo + window)
        V = np.vander(t[sl] - t[i], order + 1, increasing=True)
        out[i] = np.linalg.lstsq(V, values[sl], rcond=None)[0][0]
    return out


def smooth(t, values, ok, window=SG_WINDOW, order=SG_ORDER):
    """Savitzky-Golay filter applied separately to every unbroken run; a run shorter than the window gets a shorter one."""
    out = values.copy()
    for a, b in segments(ok):
        w = min(window, b - a if (b - a) % 2 else b - a - 1)
        if w > order:
            out[a:b] = savgol_times(t[a:b], values[a:b], w, order)
    return out


def resample(t, values, ok, hz=CONTROL_HZ):
    """Linear resampling onto a uniform grid from the first to the last valid frame; NaN inside long gaps."""
    idx = np.flatnonzero(ok)
    grid = t[idx[0]] + np.arange(int(np.floor((t[idx[-1]] - t[idx[0]]) * hz + 1e-9)) + 1) / hz
    out = np.full((len(grid), values.shape[1]), np.nan)
    for a, b in segments(ok):
        inside = (grid >= t[a] - 1e-9) & (grid <= t[b - 1] + 1e-9)
        for c in range(values.shape[1]):
            out[inside, c] = np.interp(grid[inside], t[a:b], values[a:b, c])
    return grid, out


def contact_mask(t, xy, speed_thr=CONTACT_SPEED, min_run_s=CONTACT_MIN_RUN_S, fill_s=CONTACT_FILL_S):
    """Boolean per sample: the cube is being pushed (speed above threshold), pauses < fill_s bridged, blips dropped."""
    moving = np.zeros(len(t), bool)
    ok = np.isfinite(xy[:, 0])
    for a, b in segments(ok):
        if b - a >= 2:
            v = np.gradient(xy[a:b], t[a:b], axis=0, edge_order=1 if b - a == 2 else 2)
            moving[a:b] = np.hypot(v[:, 0], v[:, 1]) > speed_thr
    dt = float(np.median(np.diff(t)))
    # bridge short pauses, then drop short runs
    off = ~moving
    for a, b in segments(off):
        if 0 < a and b < len(t) and (b - a) * dt < fill_s:
            moving[a:b] = True
    for a, b in segments(moving):
        if (b - a) * dt < min_run_s:
            moving[a:b] = False
    return moving


@dataclass
class Cleaned:
    t: np.ndarray             # (M,) s, uniform at the control rate, 0 at the first retained frame
    xy_yaw: np.ndarray        # (M, 3) NaN inside gaps longer than MAX_GAP frames; yaw unwrapped
    contact: np.ndarray       # (M,) bool
    t_offset: float           # s on the camera clock of t == 0
    n_dropped: int            # frames removed by the confidence gate
    n_interpolated: int       # frames filled by short-gap interpolation
    max_gap: int              # longest run of unusable frames between usable ones, before interpolation (0 if none)
    raw_resid: np.ndarray     # (3,) rms of (retained raw - smoothed) in x, y (m) and yaw (rad)


def clean_trajectory(t, values, keep, hz=CONTROL_HZ):
    """values (N, 3) = x, y, yaw per frame, keep (N,) bool from ``confidence_mask``."""
    v = values.copy()
    v[~keep] = np.nan
    v[:, 2] = unwrap_nan(v[:, 2])
    filled, ok = interpolate_gaps(t, np.nan_to_num(v), keep)
    smoothed = smooth(t, np.nan_to_num(filled), ok)
    grid, out = resample(t, smoothed, ok, hz)         # NaN only where the grid lies in an unbridged gap
    resid = np.sqrt(np.mean((v[keep] - smoothed[keep]) ** 2, axis=0))
    t_off = float(grid[0])
    contact = contact_mask(grid - t_off, out[:, :2])
    return Cleaned(grid - t_off, out, contact, t_off, int((~keep).sum()), int((ok & ~keep).sum()),
                   longest_gap_between(keep), resid)


def longest_gap_between(ok):
    """Longest run of unusable frames with a usable frame on each side."""
    idx = np.flatnonzero(ok)
    return int(np.diff(idx).max() - 1) if len(idx) > 1 else 0
