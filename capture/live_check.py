"""Live pose check before recording (C1.2): is the cube still, and does it stay detected while pushed?

    python -m capture.live_check --width 1280 --height 720

Detects the 2x2 GridBoard (DICT_4X4_50, IDs 0-3) and the cube marker (ID 10), draws both frames, and
prints the cube's (x, y, yaw) in the board frame with a rolling standard deviation. With the cube
still, the std should be about 1 mm or less; any dropout (cube or board lost after being tracked) during a push shows up in the counters.

Board frame (OpenCV GridBoard): origin at marker 0's corner, x right, y down, z into the table.
So the cube's z is negative (about -cube height) and yaw, the angle of the cube marker's x axis about
board z, is clockwise-positive seen from above.

Prop sizes come from data/props.yaml (measured values, mm). Keys: q quit | r reset the rolling window.
"""
import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from capture.common import (DEFAULT_INTRINSICS, DEFAULT_PROPS, load_intrinsics, load_props, lock_exposure,
                            open_capture, parse_source, read_frame, require_frame_size)

BOARD_IDS = (0, 1, 2, 3)
CUBE_ID = 10
MIN_BOARD_MARKERS = 2     # 8 coplanar corners; one marker alone gives a poorly conditioned board pose


def make_detector():
    params = cv2.aruco.DetectorParameters()
    # Default is no corner refinement, which costs sub-pixel accuracy. Pose jitter is the quantity checked here.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), params)


def make_board(marker_side, gap):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.GridBoard((2, 2), marker_side, gap, dictionary, np.array(BOARD_IDS))


def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = np.asarray(tvec).ravel()
    return T


def reprojection_rms(obj, img, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - np.asarray(img).reshape(-1, 2)) ** 2, axis=1))))


def board_pose(board, corners, ids, K, dist):
    """T_cam_board, number of board markers used, reprojection RMS (px); or None."""
    keep = [i for i, m in enumerate(ids.ravel()) if m in BOARD_IDS]
    if len(keep) < MIN_BOARD_MARKERS:
        return None
    obj, img = board.matchImagePoints([corners[i] for i in keep], ids[keep])
    # all board points are coplanar: IPPE is the right solver, LM then polishes it
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return rt_to_T(rvec, tvec), len(keep), reprojection_rms(obj, img, rvec, tvec, K, dist)


def marker_pose(corners, side, K, dist):
    """T_cam_marker for one detected square marker (marker frame: centre origin, x right, y up, z out)."""
    h = side / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)   # TL, TR, BR, BL
    img = np.asarray(corners, np.float32).reshape(4, 2)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return rt_to_T(rvec, tvec), reprojection_rms(obj, img, rvec, tvec, K, dist)


@dataclass
class Observation:
    x: float            # m, board frame
    y: float
    z: float            # m; about -cube height (board z points into the table)
    yaw: float          # rad, angle of the cube marker's x axis about board z
    tilt: float         # rad, angle between the cube marker's normal and the table normal (0 = lying flat)
    n_board: int
    board_err: float    # px
    cube_err: float     # px
    T_cam_board: np.ndarray
    T_cam_cube: np.ndarray


def observe(frame, detector, board, props, K, dist):
    """Detect and return (Observation or None, corners, ids). None unless board and cube are both found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, corners, ids
    bp = board_pose(board, corners, ids, K, dist)
    cube_idx = [i for i, m in enumerate(ids.ravel()) if m == CUBE_ID]
    if bp is None or not cube_idx:
        return None, corners, ids
    T_cb, n_board, board_err = bp
    cp = marker_pose(corners[cube_idx[0]], props["cube_marker_side"], K, dist)
    if cp is None:
        return None, corners, ids
    T_cc, cube_err = cp
    T = np.linalg.inv(T_cb) @ T_cc          # cube marker in the board frame
    R = T[:3, :3]
    # marker z points out of the marker; for a cube lying flat that is board -z
    tilt = float(np.arccos(np.clip(-R[2, 2], -1, 1)))
    return Observation(T[0, 3], T[1, 3], T[2, 3], float(np.arctan2(R[1, 0], R[0, 0])), tilt,
                       n_board, board_err, cube_err, T_cb, T_cc), corners, ids


class RollingStats:
    """Mean-free spread of the last ``window`` observations; yaw uses a circular mean so +-180 deg is safe."""

    def __init__(self, window=30):
        self.buf = deque(maxlen=window)

    def push(self, x, y, yaw):
        self.buf.append((x, y, yaw))

    def reset(self):
        self.buf.clear()

    def full(self):
        return len(self.buf) == self.buf.maxlen

    def std(self):
        """(std x, std y, std yaw) in m, m, rad; NaN until at least 2 samples."""
        if len(self.buf) < 2:
            return (np.nan,) * 3
        a = np.array(self.buf)
        mean_yaw = np.arctan2(np.sin(a[:, 2]).mean(), np.cos(a[:, 2]).mean())
        dyaw = np.arctan2(np.sin(a[:, 2] - mean_yaw), np.cos(a[:, 2] - mean_yaw))
        return float(a[:, 0].std()), float(a[:, 1].std()), float(dyaw.std())


def draw(frame, obs, corners, ids, props, K, dist):
    vis = frame.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    if obs is not None:
        cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(obs.T_cam_board[:3, :3])[0], obs.T_cam_board[:3, 3],
                          props["board_marker_side"])
        cv2.drawFrameAxes(vis, K, dist, cv2.Rodrigues(obs.T_cam_cube[:3, :3])[0], obs.T_cam_cube[:3, 3],
                          props["cube_marker_side"] * 0.75)
    return vis


def format_line(obs, stats, seen, total, dropouts):
    sx, sy, syaw = stats.std()
    pose = (f"x {obs.x * 1000:8.2f} y {obs.y * 1000:8.2f} mm  yaw {np.degrees(obs.yaw):8.2f} deg  "
            f"z {obs.z * 1000:7.1f} tilt {np.degrees(obs.tilt):4.1f}"
            if obs else "cube/board not both visible" + " " * 40)
    spread = ("std x %.2f y %.2f mm yaw %.2f deg" % (sx * 1000, sy * 1000, np.degrees(syaw))
              if stats.full() else f"std: filling window ({len(stats.buf)}/{stats.buf.maxlen})")
    return f"{pose} | {spread} | seen {seen}/{total} dropout events {dropouts}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index (default 0) or a video file")
    ap.add_argument("--width", type=int, help="requested frame width; must match the intrinsics")
    ap.add_argument("--height", type=int, help="requested frame height")
    ap.add_argument("--fourcc", help="pixel format request, e.g. MJPG")
    ap.add_argument("--exposure", type=float, help="manual exposure value (driver units). Default: pin auto's choice")
    ap.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS))
    ap.add_argument("--props", default=str(DEFAULT_PROPS))
    ap.add_argument("--window", type=int, default=30, help="frames in the rolling standard deviation (default 30)")
    ap.add_argument("--print-hz", type=float, default=4.0, help="terminal print rate (default 4)")
    ap.add_argument("--no-display", action="store_true", help="no window (headless); stop with --max-frames or end of video")
    ap.add_argument("--max-frames", type=int, help="stop after this many frames")
    args = ap.parse_args(argv)

    props = load_props(args.props, require=("board_marker_side", "board_gap", "cube_marker_side"))
    K, dist, intr_size = load_intrinsics(args.intrinsics)
    source = parse_source(args.source)
    cap, backend = open_capture(source, args.width, args.height, args.fourcc)
    if isinstance(source, int):
        lock_exposure(cap, backend, args.exposure)

    detector = make_detector()
    board = make_board(props["board_marker_side"], props["board_gap"])
    stats = RollingStats(args.window)
    total = seen = dropouts = 0
    was_seen = False
    last_print = 0.0
    print("board frame: origin at marker 0 corner, x right, y down, z into the table (cube z < 0); "
          "yaw clockwise-positive from above")
    while True:
        frame = read_frame(cap)
        if frame is None or (args.max_frames and total >= args.max_frames):
            break
        if total == 0:
            require_frame_size(frame, intr_size)
        total += 1
        obs, corners, ids = observe(frame, detector, board, props, K, dist)
        if obs is not None:
            seen += 1
            stats.push(obs.x, obs.y, obs.yaw)
        elif was_seen:
            dropouts += 1                      # a frame lost after the cube was being tracked
        was_seen = obs is not None
        if time.time() - last_print >= 1 / args.print_hz:
            print(format_line(obs, stats, seen, total, dropouts))
            last_print = time.time()
        if not args.no_display:
            cv2.imshow("live_check", draw(frame, obs, corners, ids, props, K, dist))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                stats.reset()
                seen = total = dropouts = 0
                print("window and counters reset")
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"\nframes {total}, cube+board seen {seen}, dropout events {dropouts}")
    return stats


if __name__ == "__main__":
    main()
