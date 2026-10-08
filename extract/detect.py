"""Per-frame marker detection for one recorded episode: the raw input of the extraction pipeline.

Everything here is in camera coordinates, nothing is cleaned or interpolated. The poses reuse
``capture.live_check`` (same detector settings, board and marker maths as the live check), so a pose
read from a saved clip is the pose the live check showed.

Board poses are only kept for frames where all four board markers are detected: with two markers the
board depth wanders 10-25 mm with no real motion (C1.3 QA). Frames with fewer are NaN here and are
filled later (extract.table).
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from capture.live_check import BOARD_IDS, CUBE_ID, board_pose, make_board, make_detector, marker_pose

N_BOARD = len(BOARD_IDS)


@dataclass
class RawEpisode:
    t: np.ndarray               # (N,) s from the first frame, from the CSV timestamps
    n_board: np.ndarray         # (N,) board markers detected
    T_cam_board: np.ndarray     # (N, 4, 4) NaN unless n_board == 4
    board_rms: np.ndarray       # (N,) px reprojection RMS of the board pose
    board_px: np.ndarray        # (N, 4, 2) board marker centres in the image (NaN if not detected), order of BOARD_IDS
    T_cam_cube: np.ndarray      # (N, 4, 4) NaN where the cube marker is not detected
    cube_rms: np.ndarray        # (N,) px
    cube_corners: np.ndarray    # (N, 4, 2) cube marker corners in the image (TL, TR, BR, BL), NaN if not detected

    def __len__(self):
        return len(self.t)

    def save(self, path):
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def read_timestamps(csv_path):
    """Per-frame ``perf_counter`` timestamps of an episode, seconds from the first frame."""
    rows = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
    return rows[:, 1] - rows[0, 1]


def detect_frame(gray, detector, board, props, K, dist):
    """Poses in one frame: (n_board, T_cam_board | None, board_rms, board_px, T_cam_cube | None, cube_rms, cube_corners)."""
    corners, ids, _ = detector.detectMarkers(gray)
    board_px = np.full((N_BOARD, 2), np.nan)
    if ids is None:
        return 0, None, np.nan, board_px, None, np.nan, np.full((4, 2), np.nan)
    flat = ids.ravel()
    for k, mid in enumerate(BOARD_IDS):
        hit = np.flatnonzero(flat == mid)
        if len(hit):
            board_px[k] = np.asarray(corners[hit[0]]).reshape(4, 2).mean(axis=0)
    n_board = int(np.isfinite(board_px[:, 0]).sum())
    bp = board_pose(board, corners, ids, K, dist) if n_board == N_BOARD else None
    T_board, board_rms = (bp[0], bp[2]) if bp else (None, np.nan)
    cube_idx = np.flatnonzero(flat == CUBE_ID)
    T_cube, cube_rms, cube_corners = None, np.nan, np.full((4, 2), np.nan)
    if len(cube_idx):
        cube_corners = np.asarray(corners[cube_idx[0]]).reshape(4, 2)
        cp = marker_pose(cube_corners, props["cube_marker_side"], K, dist)
        if cp:
            T_cube, cube_rms = cp
    return n_board, T_board, board_rms, board_px, T_cube, cube_rms, cube_corners


def detect_episode(video_path, csv_path, props, K, dist, size=None):
    """Run detection over every frame of ``video_path``. Frames and CSV rows must match one to one."""
    t = read_timestamps(csv_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video_path}")
    detector = make_detector()
    board = make_board(props["board_marker_side"], props["board_gap"])
    n = len(t)
    nan44 = np.full((4, 4), np.nan)
    out = RawEpisode(t=t, n_board=np.zeros(n, int), T_cam_board=np.full((n, 4, 4), np.nan),
                     board_rms=np.full(n, np.nan), board_px=np.full((n, N_BOARD, 2), np.nan),
                     T_cam_cube=np.full((n, 4, 4), np.nan), cube_rms=np.full(n, np.nan),
                     cube_corners=np.full((n, 4, 2), np.nan))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= n:
            raise ValueError(f"{video_path}: more video frames than the {n} rows of {csv_path}")
        if size is not None and (frame.shape[1], frame.shape[0]) != tuple(size):
            raise ValueError(f"{video_path}: frames are {frame.shape[1]}x{frame.shape[0]}, intrinsics are for {size}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nb, Tb, brms, bpx, Tc, crms, ccor = detect_frame(gray, detector, board, props, K, dist)
        out.n_board[i] = nb
        out.T_cam_board[i] = nan44 if Tb is None else Tb
        out.board_rms[i] = brms
        out.board_px[i] = bpx
        out.T_cam_cube[i] = nan44 if Tc is None else Tc
        out.cube_rms[i] = crms
        out.cube_corners[i] = ccor
        i += 1
    cap.release()
    if i != n:
        raise ValueError(f"{video_path}: {i} video frames but {n} rows in {csv_path}")
    return out


def episode_paths(raw_dir, ep_id):
    raw_dir = Path(raw_dir)
    return raw_dir / f"{ep_id}.avi", raw_dir / f"{ep_id}_t.csv"
