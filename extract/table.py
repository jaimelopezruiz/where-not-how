"""Table frame and cube pose (C2.1, C2.2).

Table frame: z up, right-handed, in metres.
    origin  the board's origin (corner of marker 0), moved along the table normal onto the table plane
    x       the board's x axis projected onto the table plane
    y       z cross x (the board's y axis, which points down the sheet, flipped: image-up on the sheet)
    yaw     counter-clockwise positive seen from above, the direction of the cube marker's x axis
This differs from OpenCV's GridBoard frame (x right, y down, z into the table), where yaw is clockwise.

Three things keep the board's weak orientation estimate out of the data:
  * the camera pose in the board frame is estimated per frame, from frames with all four board markers
    only, and held or interpolated across occlusions (the camera is not fixed between episodes and moves
    within a few);
  * the table normal is not taken from the board but fitted to the cube marker positions, because the
    cube always lies flat;
  * the board is used for the origin and the x direction only.

A straight push gives points along one line, which cannot fix the plane's tilt across that line. The fit
is therefore ridge-regularised towards a prior slope (the median of the episodes whose points do span the
plane): directions the data spreads over fix the plane, the others fall back on the prior.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

BOARD_RMS_MAX = 1.2      # px. 4-marker board poses sit at 0.72-0.78 px; spikes to 2-5 px are partly hidden markers
MIN_SPREAD = 0.025       # m. Plane points must spread over at least this (1 sigma, minor axis) to set the prior
PRIOR_SPREAD = 0.015     # m. Spread below which the prior outweighs the data in that direction


# ---------------------------------------------------------------- board pose per frame

@dataclass
class BoardTrack:
    T_cam_board: np.ndarray    # (N, 4, 4) every frame, measured or filled
    measured: np.ndarray       # (N,) bool: the pose comes from a clean 4-marker detection in this frame
    longest_hold_s: float      # longest stretch of unmeasured frames, in seconds


def measured_board_mask(n_board, board_rms, T_cam_board):
    return (n_board == 4) & (board_rms <= BOARD_RMS_MAX) & np.isfinite(T_cam_board[:, 0, 0])


def fill_board_pose(t, T_cam_board, measured):
    """Interpolate (slerp + linear) between measured frames and hold the nearest one at the ends."""
    idx = np.flatnonzero(measured)
    if len(idx) < 2:
        raise ValueError(f"{len(idx)} frames with a clean 4-marker board pose; need at least 2")
    tq = np.clip(t, t[idx[0]], t[idx[-1]])
    slerp = Slerp(t[idx], Rotation.from_matrix(T_cam_board[idx, :3, :3]))
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3] = slerp(tq).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(tq, t[idx], T_cam_board[idx, k, 3])
    out[idx] = T_cam_board[idx]
    return out


def longest_run(mask):
    """Longest run of True in a boolean array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def board_track(raw):
    measured = measured_board_mask(raw.n_board, raw.board_rms, raw.T_cam_board)
    filled = fill_board_pose(raw.t, raw.T_cam_board, measured)
    unmeasured = ~measured
    hold_s, run_start = 0.0, None
    for i, u in enumerate(np.append(unmeasured, False)):     # sentinel closes a run at the end
        if u and run_start is None:
            run_start = i
        elif not u and run_start is not None:
            lo = raw.t[run_start - 1] if run_start > 0 else raw.t[run_start]
            hi = raw.t[i] if i < len(raw.t) else raw.t[i - 1]
            hold_s = max(hold_s, float(hi - lo))
            run_start = None
    return BoardTrack(filled, measured, hold_s)


def board_pixel_drift(board_px, measured):
    """Board marker centroid movement in the image, in px: (first to last clean frame, max from the first)."""
    idx = np.flatnonzero(measured)
    c = board_px[idx].mean(axis=1)                  # (K, 2) centroid of the four marker centres
    return float(np.linalg.norm(c[-1] - c[0])), float(np.linalg.norm(c - c[0], axis=1).max())


def board_hold_deviation(board_px, measured, t):
    """95th percentile pixel offset of partly visible board markers from where the held track says they are.

    Frames with 1-3 board markers cannot give a pose but do show whether the camera moved while the
    board was hidden. The reference is each marker's centre interpolated between clean frames. A
    percentile, not the max: a half-covered marker's corners are biased for a few frames at the edges of
    an occlusion, which would dominate a max (sustained offsets of a static camera sit near 1 px).
    """
    idx = np.flatnonzero(measured)
    dev = []
    for k in range(board_px.shape[1]):
        seen = np.flatnonzero(np.isfinite(board_px[:, k, 0]) & ~measured)
        if len(seen) == 0:
            continue
        tq = np.clip(t[seen], t[idx[0]], t[idx[-1]])
        ref = np.stack([np.interp(tq, t[idx], board_px[idx, k, a]) for a in range(2)], axis=1)
        dev.append(np.linalg.norm(board_px[seen, k] - ref, axis=1))
    return float(np.percentile(np.concatenate(dev), 95)) if dev else 0.0


# ---------------------------------------------------------------- table plane

@dataclass
class Plane:
    """Plane z = a + b x + c y in the board frame (board z points into the table)."""
    a: float
    b: float
    c: float
    resid_rms: float        # m, distance of the fitted points from the plane
    resid_max: float        # m
    spread: float           # m, 1 sigma of the points along the weakest in-plane direction
    prior_weight: float     # 0..1, share of the prior in the weakest direction (0 = data alone)

    @property
    def normal_up(self):
        n = np.array([self.b, self.c, -1.0])
        return n / np.linalg.norm(n)

    @property
    def height(self):
        """Height of the plane above the board origin, along the normal (about the cube's height)."""
        return float(self.normal_up @ np.array([0.0, 0.0, self.a]))

    @property
    def tilt_deg(self):
        return float(np.degrees(np.arccos(-self.normal_up[2])))

    @property
    def slope(self):
        return np.array([self.b, self.c])

    def axes(self):
        """Rows are the table x, y, z axes in board coordinates (z up, x = board x projected onto the plane)."""
        z = self.normal_up
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(z, x), z])


def fit_plane(points, prior=None, spread_scale=PRIOR_SPREAD):
    """Least-squares plane through marker positions (board frame), optionally ridge-pulled to ``prior``=(b, c)."""
    p = np.asarray(points, float)
    mean = p.mean(axis=0)
    q = p - mean
    cov = q[:, :2].T @ q[:, :2] / len(p)               # in-plane covariance (m^2)
    cross = q[:, :2].T @ q[:, 2] / len(p)
    lam = np.linalg.eigvalsh(cov)
    mu = 0.0 if prior is None else spread_scale ** 2
    prior_vec = np.zeros(2) if prior is None else np.asarray(prior, float)
    slope = np.linalg.solve(cov + mu * np.eye(2) + 1e-12 * np.eye(2), cross + mu * prior_vec)
    a = mean[2] - slope @ mean[:2]
    plane = Plane(float(a), float(slope[0]), float(slope[1]), 0.0, 0.0, float(np.sqrt(max(lam[0], 0.0))),
                  float(mu / (lam[0] + mu)) if mu else 0.0)
    d = (p - np.array([0.0, 0.0, a])) @ plane.normal_up
    plane.resid_rms = float(np.sqrt(np.mean(d ** 2)))
    plane.resid_max = float(np.abs(d).max())
    return plane


def prior_slope(planes):
    """Median slope of the planes whose points spread enough to have fixed it themselves; None if none do."""
    good = [pl.slope for pl in planes if pl.spread >= MIN_SPREAD]
    return np.median(good, axis=0) if good else None


# ---------------------------------------------------------------- cube pose

def cube_in_board(T_cam_board, T_cam_cube):
    """Cube marker pose in the board frame, (N, 4, 4); NaN where the marker was not detected."""
    out = np.full_like(T_cam_cube, np.nan)
    ok = np.isfinite(T_cam_cube[:, 0, 0])
    out[ok] = np.linalg.inv(T_cam_board[ok]) @ T_cam_cube[ok]
    return out


def cube_to_table(plane, T_board_cube):
    """(x, y, yaw, tilt, z): marker pose in the table frame. NaN rows stay NaN.

    tilt is the angle between the marker's normal and the table normal (0 for a box lying flat);
    z is the marker's height above the fitted plane (0 by construction on average).
    """
    A = plane.axes()                                       # board -> table rotation
    R = A @ T_board_cube[:, :3, :3]
    pos = T_board_cube[:, :3, 3] @ A.T                     # x, y are measured from the board origin's foot on the plane
    pos[:, 2] -= plane.height
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    tilt = np.arccos(np.clip(R[:, 2, 2], -1, 1))
    return pos[:, 0], pos[:, 1], yaw, tilt, pos[:, 2]


def recovered_marker_side(plane, T_cam_board, corners, K, dist):
    """Median cube marker edge length (m) measured from the image alone: each corner's ray against the plane.

    The pose solver is given the marker's side, so its output cannot test it. Intersecting the corner rays
    with the fitted plane (at marker height) only uses the board's scale, so this checks that the board and
    the marker prints agree with ``props.yaml``. Needs the plane at marker height, which is what ``fit_plane`` returns.
    """
    ok = np.flatnonzero(np.isfinite(corners[:, 0, 0]))
    n_up = plane.normal_up
    sides = []
    for i in ok:
        R, t = T_cam_board[i, :3, :3], T_cam_board[i, :3, 3]
        n_cam = R @ n_up
        d_cam = plane.height + n_cam @ t                   # n.p = d in the board frame, moved to the camera frame
        rays = cv2.undistortPoints(corners[i].reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
        rays = np.column_stack([rays, np.ones(len(rays))])
        pts = rays * (d_cam / (rays @ n_cam))[:, None]
        sides.append(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1).mean())
    return float(np.median(sides)) if sides else float("nan")
