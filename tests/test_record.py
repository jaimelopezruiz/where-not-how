"""capture/record.py against a fake camera feeding rendered board + cube frames.

Run from the repo root:  python -m tests.test_record   (or: python -m pytest tests/test_record.py)

The fake camera hands out one frame per loop iteration and a scripted key per iteration. record.py needs
10 frames to measure fps before it will record, so the first WARMUP iterations are copies of scene 0. A key
pressed on iteration k is handled after frame k was read: with SPACE (start) on iteration WARMUP and SPACE
(stop) on iteration WARMUP + 9, the recorded frames are iterations WARMUP + 1 to WARMUP + 9, i.e. scenes 1 to 9.
"""
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from capture import record
from tests.test_calibrate import FakeCap
from tests.test_live_check import BOARD, DETECTOR, DIST, K, PROPS, SIZE, scene

N_FRAMES = 12
WARMUP = record.MIN_FPS_SAMPLES
START_ITER, STOP_ITER = WARMUP, WARMUP + 9
SPACE = ord(" ")
_FRAMES = []


def frames():
    """Twelve 1280x720 scenes with the cube drifting and turning, rendered once."""
    if not _FRAMES:
        _FRAMES.extend(scene(0.28 + 0.004 * i, 0.12, np.radians(10 + 2 * i), noise=1.0, seed=i)[0] for i in range(N_FRAMES))
    return _FRAMES


def write_inputs(tmp, size=SIZE):
    """Intrinsics for the fake camera and a props file, in tmp."""
    np.savez(tmp / "intr.npz", camera_matrix=K, dist_coeffs=DIST, image_size=np.array(size))
    (tmp / "props.yaml").write_text("board: {marker_side_mm: 60, gap_mm: 40}\ncube: {marker_side_mm: 40}\n")


def run(tmp, keys, extra=(), feed=None):
    """Run record.main on the fake camera with scripted keys; returns what main returns."""
    feed = [frames()[0]] * WARMUP + frames() if feed is None else feed
    key_iter = iter(keys)
    orig = (record.open_capture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, record.lock_exposure)
    record.open_capture = lambda *a, **k: (FakeCap(feed), cv2.CAP_DSHOW)     # real cv2.VideoCapture stays for read-back
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: next(key_iter, -1)
    cv2.destroyAllWindows = lambda: None
    record.lock_exposure = lambda *a, **k: -6.0
    try:
        return record.main(["--source", "0", "--intrinsics", str(tmp / "intr.npz"), "--props", str(tmp / "props.yaml"),
                            "--out-dir", str(tmp / "raw"), "--manifest", str(tmp / "manifest.csv"),
                            "--notes", "index finger, slow", *extra])
    finally:
        record.open_capture, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows, record.lock_exposure = orig


def one_episode_keys(category="s"):
    keys = [-1] * (STOP_ITER + 2)
    keys[0], keys[START_ITER], keys[STOP_ITER], keys[STOP_ITER + 1] = ord(category), SPACE, SPACE, ord("q")
    return keys


def read_video(path):
    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def test_episode_files_are_consistent():
    """Frames lost or duplicated between the video, the timestamp CSV and the camera would misalign extraction."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_inputs(tmp)
        (summary,) = run(tmp, one_episode_keys("s"))
        n = STOP_ITER - START_ITER                      # scenes 1..9
        raw = tmp / "raw"
        video = read_video(raw / "ep_000.avi")
        with open(raw / "ep_000_t.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        ts = [float(r["perf_counter_s"]) for r in rows]
        assert len(video) == len(rows) == summary["frames"] == summary["video_frames"] == n
        assert [int(r["frame_index"]) for r in rows] == list(range(n))
        assert all(b > a for a, b in zip(ts, ts[1:]))   # strictly increasing
        with open(tmp / "manifest.csv", newline="") as f:
            assert list(csv.reader(f)) == [["id", "category", "notes"], ["ep_000", "straight", "index finger, slow"]]
        meta = json.loads((raw / "ep_000_meta.json").read_text())
        assert meta["frame_size"] == list(SIZE) and meta["frames"] == n
        assert meta["intrinsics_sha256"] == hashlib.sha256((tmp / "intr.npz").read_bytes()).hexdigest()
        assert meta["writer"] == {"backend": "CV_MJPEG", "codec": "MJPG", "quality": 100}
        assert meta["props"]["cube"]["marker_side_mm"] == 40


def test_recorded_video_gives_the_same_pose_as_the_source():
    """A lossy or low-quality writer shifts detected marker corners, so the cube pose from the file would be biased."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_inputs(tmp)
        run(tmp, one_episode_keys("c"))
        video = read_video(tmp / "raw" / "ep_000.avi")
    worst = np.zeros(3)
    for i, back in enumerate(video):
        src = frames()[1 + i]                           # recorded frame i is scene 1 + i
        a = record_obs(src)
        b = record_obs(back)
        worst = np.maximum(worst, [abs(a[0] - b[0]), abs(a[1] - b[1]), abs(np.arctan2(np.sin(a[2] - b[2]), np.cos(a[2] - b[2])))])
    print(f"\nread-back vs source pose: x {worst[0] * 1000:.3f} mm, y {worst[1] * 1000:.3f} mm, yaw {np.degrees(worst[2]):.4f} deg")
    assert worst[0] < 3e-4 and worst[1] < 3e-4 and np.degrees(worst[2]) < 0.05    # measured max: 0.11 mm, 0.01 deg


def record_obs(frame):
    from capture import live_check
    o, _, _ = live_check.observe(frame, DETECTOR, BOARD, PROPS, K, DIST)
    assert o is not None
    return o.x, o.y, o.yaw


def test_wrong_frame_size_is_refused_before_any_file_is_written():
    """Recording at a resolution other than the intrinsics' would make every extracted pose silently wrong."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_inputs(tmp, size=(640, 480))
        try:
            run(tmp, one_episode_keys("s"))
        except SystemExit as e:
            assert "640x480" in str(e)
        else:
            raise AssertionError("size mismatch was accepted")
        assert not (tmp / "raw").exists() and not (tmp / "manifest.csv").exists()


def test_existing_episode_is_never_overwritten():
    """A second session reusing episode numbers would destroy recorded data."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_inputs(tmp)
        raw = tmp / "raw"
        raw.mkdir()
        (raw / "ep_000.avi").write_bytes(b"precious")
        (tmp / "manifest.csv").write_text("id,category,notes\nep_000,straight,earlier session\n")
        (summary,) = run(tmp, one_episode_keys("t"))
        assert summary["id"] == "ep_001"
        assert (raw / "ep_000.avi").read_bytes() == b"precious"
        rows = (tmp / "manifest.csv").read_text().splitlines()
        assert rows[-1].startswith("ep_001,turn") and len(rows) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
