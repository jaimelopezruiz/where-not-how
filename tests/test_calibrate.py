"""capture/calibrate.py against synthetic checkerboard views with known intrinsics.

Run from the repo root:  python -m tests.test_calibrate
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from capture import calibrate
from tests import synth

SIZE = (1280, 720)
PATTERN = (9, 6)
SQUARE = 0.025
K_TRUE = synth.make_K(*SIZE)
DIST_TRUE = np.array([-0.04, 0.01, 0.0005, -0.0003, 0.0])


def random_views(n, seed=1):
    """n camera poses looking at the board with random distance, tilt and roll, board fully in view."""
    rng = np.random.default_rng(seed)
    centre = np.array([(PATTERN[0] - 1) * SQUARE / 2, (PATTERN[1] - 1) * SQUARE / 2, 0])
    poses = []
    while len(poses) < n:
        d = rng.uniform(0.30, 0.55)
        tilt = np.radians(rng.uniform(0, 38))
        az = rng.uniform(0, 2 * np.pi)
        pos = centre + np.array([d * np.sin(tilt) * np.cos(az), d * np.sin(tilt) * np.sin(az), -d * np.cos(tilt)])
        roll = np.radians(rng.uniform(-30, 30))
        down = np.array([-np.sin(roll), np.cos(roll), 0])
        T = synth.look_at(pos, centre + rng.uniform(-0.03, 0.03, 3) * [1, 1, 0], down)
        if synth.checkerboard_in_view(K_TRUE, T, SIZE, PATTERN, SQUARE):
            poses.append(T)
    return poses


def render(T, seed=0):
    return synth.render_checkerboard(K_TRUE, DIST_TRUE, T, SIZE, PATTERN, SQUARE, noise=1.5, seed=seed)


def test_recovers_known_intrinsics():
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp) / "frames"
        frames.mkdir()
        for i, T in enumerate(random_views(25)):
            cv2.imwrite(str(frames / f"f{i:02d}.png"), render(T, seed=i))
        out = Path(tmp) / "intrinsics.npz"
        calibrate.main(["--square-mm", str(SQUARE * 1000), "--from-frames", str(frames), "--out", str(out)])
        with np.load(out) as f:
            got = {k: f[k] for k in f.files}
    K, dist = got["camera_matrix"], got["dist_coeffs"]
    print(f"\ntruth    fx={K_TRUE[0, 0]:.1f} fy={K_TRUE[1, 1]:.1f} cx={K_TRUE[0, 2]:.1f} cy={K_TRUE[1, 2]:.1f} k1={DIST_TRUE[0]}")
    print(f"recovered fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} k1={dist[0]:.4f}")
    assert tuple(got["image_size"]) == SIZE
    assert abs(K[0, 0] / K_TRUE[0, 0] - 1) < 0.01 and abs(K[1, 1] / K_TRUE[1, 1] - 1) < 0.01
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 3 and abs(K[1, 2] - K_TRUE[1, 2]) < 3
    assert abs(dist[0] - DIST_TRUE[0]) < 0.02
    assert got["rms"] < calibrate.ACCEPT_RMS_PX
    assert abs(float(got["square_m"]) - SQUARE) < 1e-12
    # per-view errors must agree with the overall RMS (guards the corner-shape broadcast bug)
    assert abs(np.sqrt(np.mean(got["per_view_rms"] ** 2)) - got["rms"]) < 0.05


def test_fix_k3_holds_k3_at_zero():
    found = [calibrate.find_corners(render(T, i), PATTERN) for i, T in enumerate(random_views(15, seed=3))]
    corners = [c for ok, c in found if ok]          # a steep view may legitimately fail to detect
    assert len(corners) >= 10
    assert calibrate.calibrate(corners, PATTERN, SQUARE, SIZE, fix_k3=True)["dist_coeffs"][4] == 0.0


def test_blank_image_has_no_board():
    found, corners = calibrate.find_corners(np.full((SIZE[1], SIZE[0]), 200, np.uint8), PATTERN)
    assert not found and corners is None


def test_offline_refuses_too_few_frames():
    with tempfile.TemporaryDirectory() as tmp:
        for i, T in enumerate(random_views(3)):
            cv2.imwrite(str(Path(tmp) / f"f{i}.png"), render(T))
        try:
            calibrate.main(["--square-mm", "25", "--from-frames", tmp, "--out", str(Path(tmp) / "x.npz")])
        except SystemExit as e:
            assert "need at least" in str(e)
        else:
            raise AssertionError("expected SystemExit")


class FakeCap:
    """Hands out a fixed list of frames, then repeats the last one."""

    def __init__(self, frames):
        self.frames, self.i = frames, 0

    def isOpened(self):
        return True

    def read(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return True, f

    def get(self, prop):
        return 0.0

    def set(self, prop, value):
        return True

    def release(self):
        pass


def test_live_saves_only_when_board_found():
    """SPACE is pressed on every frame; only frames with a detected board may be saved."""
    n_boards = 12
    board_frames = [cv2.cvtColor(render(T, seed=i), cv2.COLOR_GRAY2BGR) for i, T in enumerate(random_views(n_boards, seed=5))]
    blank = np.full((SIZE[1], SIZE[0], 3), 180, np.uint8)
    frames, expected = [], 0
    for b in board_frames:
        frames += [blank, b]                 # blank frame first, then a board frame
    keys = [ord(" ")] * len(frames) + [ord("c"), ord("q")]
    frames += [board_frames[-1]] * 2         # frames consumed by the 'c' and 'q' iterations

    with tempfile.TemporaryDirectory() as tmp:
        out, frame_dir = Path(tmp) / "i.npz", Path(tmp) / "frames"
        saved_patches = {}
        orig = (cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, calibrate.lock_exposure)
        key_iter = iter(keys)
        cv2.VideoCapture = lambda *a, **k: FakeCap(frames)
        cv2.imshow = lambda *a, **k: None
        cv2.waitKey = lambda *a, **k: next(key_iter)
        cv2.destroyAllWindows = lambda: None
        calibrate.lock_exposure = lambda *a, **k: saved_patches.setdefault("locked", True)
        try:
            result = calibrate.main(["--square-mm", "25", "--width", str(SIZE[0]), "--height", str(SIZE[1]),
                                     "--out", str(out), "--frames-dir", str(frame_dir)])
        finally:
            cv2.VideoCapture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, calibrate.lock_exposure = orig
        n_saved = len(list(frame_dir.glob("*.png")))
        assert out.exists()
    assert saved_patches.get("locked"), "exposure lock was not attempted for a camera source"
    print(f"\nlive loop: {n_boards} board frames + {n_boards} blank, SPACE on all -> {n_saved} saved, rms {result['rms']:.3f}")
    assert n_saved == n_boards
    assert len(result["per_view_rms"]) == n_boards


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
