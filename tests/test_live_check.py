"""capture/live_check.py against synthetic board + cube scenes with known ground truth.

Run from the repo root:  python -m tests.test_live_check
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from capture import common, live_check
from tests import synth

SIZE = (1280, 720)
K = synth.make_K(*SIZE)
DIST = np.array([-0.04, 0.01, 0.0005, -0.0003, 0.0])
MARKER, GAP, CUBE_MARKER, CUBE_SIDE = 0.060, 0.040, 0.040, 0.045
PROPS = {"board_marker_side": MARKER, "board_gap": GAP, "cube_marker_side": CUBE_MARKER}
# camera ~0.5 m above the table, on the +y side of the sheet, looking down at about 45 deg at the workspace
T_CAM_BOARD = synth.look_at([0.18, 0.52, -0.48], [0.18, 0.10, 0.0], [0, 1, 0])

DETECTOR = live_check.make_detector()
BOARD = live_check.make_board(MARKER, GAP)


def scene(x, y, yaw, **kw):
    img, T_truth = synth.render_scene(K, DIST, T_CAM_BOARD, SIZE, MARKER, GAP, CUBE_MARKER, (x, y, yaw), CUBE_SIDE, **kw)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), T_truth


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def project(points, T_cam_frame):
    cam = (T_cam_frame[:3, :3] @ points.T).T + T_cam_frame[:3, 3]
    pix = (K @ cam.T).T
    return (pix[:, :2] / pix[:, 2:3]).astype(np.float32)


def test_pose_maths_is_exact_on_exact_corners():
    """Feed the estimator the analytically projected corners: any error left is a convention or maths bug."""
    pose = (0.30, 0.15, np.radians(30))
    _, T_bm = synth.render_scene(K, None, T_CAM_BOARD, SIZE, MARKER, GAP, CUBE_MARKER, pose, CUBE_SIDE)
    h = CUBE_MARKER / 2
    cube_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    corners = [project(o.astype(float), T_CAM_BOARD).reshape(1, 4, 2) for o in BOARD.getObjPoints()]
    corners.append(project(cube_pts, T_CAM_BOARD @ T_bm).reshape(1, 4, 2))
    ids = np.array([[0], [1], [2], [3], [10]])
    T_cb, n, _ = live_check.board_pose(BOARD, corners[:4], ids[:4], K, np.zeros(5))
    T_cc, _ = live_check.marker_pose(corners[4], CUBE_MARKER, K, np.zeros(5))
    T = np.linalg.inv(T_cb) @ T_cc
    assert n == 4
    assert np.abs(T[:3, 3] - [pose[0], pose[1], -CUBE_SIDE]).max() < 1e-6          # metres
    assert abs(wrap(np.arctan2(T[1, 0], T[0, 0]) - pose[2])) < 1e-6


# Measured on these renders: the detector's corners scatter about 0.25 px rms, which at this oblique
# viewing angle turns into up to ~4 mm in y and z (position along the line of sight is the weak
# direction) and ~0.1 deg in yaw. x is better. The exact-corner test above pins the maths itself.
POS_TOL, Z_TOL, YAW_TOL_DEG = 5e-3, 5e-3, 0.5


def test_pose_matches_ground_truth():
    cases = [(0.28, 0.05, 0.0), (0.30, 0.15, np.radians(30)), (0.25, 0.25, np.radians(-60)),
             (0.35, 0.10, np.radians(90)), (0.22, 0.20, np.radians(135))]
    worst = np.zeros(4)
    for x, y, yaw in cases:
        frame, _ = scene(x, y, yaw, noise=1.0)
        obs, _, _ = live_check.observe(frame, DETECTOR, BOARD, PROPS, K, DIST)
        assert obs is not None, f"not detected at {(x, y, yaw)}"
        err = np.array([abs(obs.x - x), abs(obs.y - y), abs(obs.z + CUBE_SIDE), abs(wrap(obs.yaw - yaw))])
        worst = np.maximum(worst, err)
        assert obs.tilt < np.radians(1.5), f"tilt={np.degrees(obs.tilt)}"
        assert obs.n_board == 4
    print(f"\nworst error over {len(cases)} poses: x {worst[0] * 1000:.2f} mm, y {worst[1] * 1000:.2f} mm, "
          f"z {worst[2] * 1000:.2f} mm, yaw {np.degrees(worst[3]):.3f} deg")
    assert worst[0] < POS_TOL and worst[1] < POS_TOL and worst[2] < Z_TOL and np.degrees(worst[3]) < YAW_TOL_DEG


def test_yaw_wraps_around_180():
    for deg in (170, 179, 181, -179, -170):
        frame, _ = scene(0.28, 0.12, np.radians(deg))
        obs, _, _ = live_check.observe(frame, DETECTOR, BOARD, PROPS, K, DIST)
        assert obs is not None
        assert abs(wrap(obs.yaw - np.radians(deg))) < np.radians(0.5), (deg, np.degrees(obs.yaw))


def test_still_cube_jitter_is_small():
    stats = live_check.RollingStats(20)
    for seed in range(20):
        frame, _ = scene(0.30, 0.12, np.radians(20), noise=2.0, seed=seed)
        obs, _, _ = live_check.observe(frame, DETECTOR, BOARD, PROPS, K, DIST)
        stats.push(obs.x, obs.y, obs.yaw)
    sx, sy, syaw = stats.std()
    print(f"\nstill-cube std over 20 noisy frames: x {sx * 1000:.3f} mm, y {sy * 1000:.3f} mm, yaw {np.degrees(syaw):.3f} deg")
    assert stats.full() and sx < 3e-4 and sy < 3e-4 and np.degrees(syaw) < 0.3


def test_rolling_std_handles_yaw_wrap():
    s = live_check.RollingStats(10)
    for d in (179.5, -179.5, 179.8, -179.8, 180.0, 179.0, -179.0, 179.6, -179.6, 179.9):
        s.push(0.0, 0.0, np.radians(d))
    assert np.degrees(s.std()[2]) < 1.0       # a naive std would be ~180 deg
    s.reset()
    assert not s.full() and np.isnan(s.std()[0])


def test_missing_cube_or_board_gives_no_observation():
    frame, _ = scene(0.3, 0.1, 0.0, with_cube=False)
    assert live_check.observe(frame, DETECTOR, BOARD, PROPS, K, DIST)[0] is None
    assert live_check.observe(np.full((SIZE[1], SIZE[0], 3), 255, np.uint8), DETECTOR, BOARD, PROPS, K, DIST)[0] is None


def test_one_board_marker_is_not_enough():
    frame, _ = scene(0.3, 0.1, 0.0)
    corners, ids, _ = DETECTOR.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    first = [i for i, m in enumerate(ids.ravel()) if m == 0]
    assert live_check.board_pose(BOARD, [corners[first[0]]], ids[first], K, DIST) is None
    assert live_check.board_pose(BOARD, corners, ids, K, DIST) is not None   # cube id 10 is ignored


def test_refuses_null_props():
    """data/props.yaml ships with null values; the scripts must refuse them, not substitute nominals."""
    try:
        common.load_props(common.DEFAULT_PROPS, require=("board_marker_side",))
    except SystemExit as e:
        assert "marker_side_mm" in str(e)
    else:
        raise AssertionError("null props were accepted")


def test_cli_on_video_and_size_check():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "props.yaml").write_text(
            f"board: {{marker_side_mm: {MARKER * 1000}, gap_mm: {GAP * 1000}}}\n"
            f"cube: {{marker_side_mm: {CUBE_MARKER * 1000}, side_mm: {CUBE_SIDE * 1000}, mass_g: 30}}\n")
        np.savez(tmp / "intr.npz", camera_matrix=K, dist_coeffs=DIST, image_size=np.array(SIZE))
        video = tmp / "clip.avi"
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, SIZE)
        assert vw.isOpened(), "no MJPG writer in this OpenCV build"
        for i in range(40):
            frame, _ = scene(0.28 + 0.0005 * i, 0.12, np.radians(10), noise=1.0, seed=i)
            if 18 <= i < 21:
                frame[:] = 255                       # three dropout frames
            vw.write(frame)
        vw.release()
        stats = live_check.main(["--source", str(video), "--intrinsics", str(tmp / "intr.npz"),
                                 "--props", str(tmp / "props.yaml"), "--no-display", "--window", "10", "--print-hz", "1000"])
        assert stats.full()
        # intrinsics for another resolution must be rejected, not silently used
        np.savez(tmp / "wrong.npz", camera_matrix=K, dist_coeffs=DIST, image_size=np.array((640, 480)))
        try:
            live_check.main(["--source", str(video), "--intrinsics", str(tmp / "wrong.npz"),
                             "--props", str(tmp / "props.yaml"), "--no-display"])
        except SystemExit as e:
            assert "640x480" in str(e)
        else:
            raise AssertionError("size mismatch was accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
