"""Tests for scripts.showcase (C8.1): timeline, crops, orientation, feasibility, GIF writer, and the pipeline
end to end on one showcase episode with a tiny step budget and a synthetic video (no data/raw needed).

Run:  python -m pytest tests/test_showcase.py
"""
import csv

import cv2
import numpy as np
import pytest

from control.workspace import WorkspaceMap
from scripts import showcase


# -- timeline -----------------------------------------------------------------------------------------------

def _trace():
    """Box at rest for 2 s, moves 0.2 m along x in 4 s, rests for 2 s (t in 0.1 s steps)."""
    t = np.arange(0, 8.0001, 0.1)
    x = np.clip((t - 2.0) / 4.0, 0, 1) * 0.2
    return np.column_stack([x, np.zeros_like(x)]), t


def test_push_window_finds_start_and_end_of_the_move():
    xy, t = _trace()
    start, end = showcase.push_window(xy, t)
    assert 2.0 < start <= 2.1 + 1e-9          # first sample more than 5 mm from the start (5 mm = 0.1 s of motion)
    assert 5.85 < end <= 6.0 + 1e-9           # box within 5 mm of where it ends from here on


def test_push_window_of_a_box_that_never_moves_is_the_whole_clip():
    t = np.arange(0, 3, 0.1)
    assert showcase.push_window(np.zeros((len(t), 2)), t) == (0.0, pytest.approx(t[-1]))


def test_gif_times_run_from_minus_lead_to_the_later_push_end_plus_tail():
    taus = showcase.gif_times((3.0, 13.0), (1.0, 6.0), lead=1.0, tail=1.5, fps=10)
    assert taus[0] == pytest.approx(-1.0)
    assert taus[-1] == pytest.approx(10.0 + 1.5)       # the longer push (10 s) sets the length
    assert np.allclose(np.diff(taus), 0.1)
    assert len(showcase.gif_times((0, 10), (0, 10), 1.0, 1.5, 10, max_seconds=2.0)) == 21


# -- crops and orientation ----------------------------------------------------------------------------------

def test_square_rect_stays_inside_the_frame():
    assert showcase.square_rect((100, 100), 50, (1280, 720)) == (50, 50, 100)
    x0, y0, s = showcase.square_rect((10, 710), 80, (1280, 720))      # near a corner: shifted, not clipped
    assert (x0, y0, s) == (0, 560, 160)
    assert showcase.square_rect((640, 360), 900, (1280, 720))[2] == 720   # too big: shrinks to the frame


def test_real_crop_covers_the_track_and_falls_back_to_the_centre():
    props = {"cube_side": 0.081, "cube_marker_side": 0.047}
    centres = np.array([[300.0, 200.0], [500.0, 600.0]])
    x0, y0, s = showcase.real_crop(centres, 88.0, (1280, 720), props)
    assert x0 < 300 and x0 + s > 500 and y0 < 200 and y0 + s > 600          # the whole track is inside
    assert showcase.real_crop(np.empty((0, 2)), np.nan, (1280, 720), props) == (280, 0, 720)


@pytest.mark.parametrize("view", ["ccw", "cw"])
def test_raw_and_sim_images_of_one_scene_agree_after_orient(view):
    """The raw frame has x right, y up; the sim render has x up, y left: a 90 deg ccw turn of each other.
    Asymmetric scene."""
    raw = np.zeros((6, 6, 3), np.uint8)
    raw[2, 5] = (255, 0, 0)        # a point on +x (right of the centre line)
    raw[0, 3] = (0, 255, 0)        # a point on +y (top)
    raw[1, 4] = (0, 0, 255)
    sim = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert np.array_equal(showcase.orient(raw, view, "raw"), showcase.orient(sim, view, "sim"))
    out = showcase.orient(raw, view, "raw")
    assert sorted(map(tuple, out.reshape(-1, 3))) == sorted(map(tuple, raw.reshape(-1, 3)))    # same pixels, turned


def test_ccw_view_puts_plus_x_up_and_plus_y_left():
    raw = np.zeros((7, 7, 3), np.uint8)
    raw[3, 6] = 255               # +x: right in the raw frame
    raw[0, 3] = 255               # +y: up in the raw frame
    out = showcase.orient(raw, "ccw", "raw")
    assert out[0, 3].all() and out[3, 0].all()       # +x is now at the top, +y at the left


# -- feasibility --------------------------------------------------------------------------------------------

def test_check_feasibility_flags_a_path_off_the_reachable_set_or_inside_the_base_radius():
    reach = np.array([[x, y] for x in np.arange(0.15, 0.40, 0.005) for y in np.arange(-0.1, 0.1, 0.005)])
    ws = WorkspaceMap(1.0, [0.0, 0.0])
    ok = np.column_stack([np.linspace(0.25, 0.35, 20), np.zeros(20)])       # pusher 4 cm behind is still in reach
    off = np.column_stack([np.linspace(0.25, 0.35, 20), np.full(20, 0.3)])  # outside the sample
    near_base = np.column_stack([np.linspace(0.05, 0.12, 20), np.zeros(20)])
    rows = showcase.check_feasibility([ok, off, near_base], ws, reach, r_min_cube=0.147, pusher_offset=0.04, tol=0.012)
    assert rows[0][1] == 1.0 and rows[1][1] == 0.0 and rows[2][1] == 0.0
    assert rows[0][0] == 20 and rows[2][2] == pytest.approx(0.05)


# -- real video panels ---------------------------------------------------------------------------------------

def _write_video(directory, ep, n=10, size=(64, 48), fps=10.0):
    """n frames, frame i filled with the grey level 20 * i (mod 256); timestamps as capture/record writes them."""
    w, h = size
    wr = cv2.VideoWriter(str(directory / f"{ep}.avi"), cv2.CAP_OPENCV_MJPEG, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    wr.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)
    for i in range(n):
        wr.write(np.full((h, w, 3), (20 * i) % 256, np.uint8))
    wr.release()
    with open(directory / f"{ep}_t.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame_index", "perf_counter_s"])
        for i in range(n):
            out.writerow([i, 1000.0 + i / fps])


def test_real_panels_pick_the_last_frame_at_or_before_each_time_and_hold_the_last(tmp_path):
    _write_video(tmp_path, "ep_x")
    times = [-1.0, 0.0, 0.05, 0.31, 0.5, 0.89, 5.0]                 # frames every 0.1 s, 10 frames
    panels = showcase.real_panels(tmp_path / "ep_x.avi", tmp_path / "ep_x_t.csv", times, (0, 0, 48), "ccw", 16)
    levels = [int(p[8, 8, 0]) for p in panels]
    assert all(p.shape == (16, 16, 3) for p in panels)
    for got, idx in zip(levels, [0, 0, 0, 3, 5, 8, 9]):
        assert abs(got - 20 * idx) <= 3                              # MJPG on flat grey: within a few levels


# -- GIF writer ----------------------------------------------------------------------------------------------

def test_save_gif_writes_every_frame_and_cuts_to_fit_the_size_limit(tmp_path):
    from PIL import Image
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (60, 80, 3), dtype=np.uint8) for _ in range(12)]
    mb, stride, colours = showcase.save_gif(frames, tmp_path / "a.gif", 10, max_mb=50)
    assert (stride, colours) == (1, 256) and Image.open(tmp_path / "a.gif").n_frames == 12
    mb2, stride2, colours2 = showcase.save_gif(frames, tmp_path / "b.gif", 10, max_mb=0.0001)   # impossible limit
    assert (stride2, colours2) == (3, 32) and Image.open(tmp_path / "b.gif").n_frames == 4


# -- the pipeline --------------------------------------------------------------------------------------------

def test_showcase_runs_end_to_end_on_one_episode_with_a_tiny_budget(tmp_path):
    """Real showcase trace and scripted pusher, 40 env steps, synthetic 'human video' (no box marker in it, so
    the crop falls back to the frame centre). Checks the files and that the sim camera really has x up, y left."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_video(raw, "ep_049", n=60, size=(320, 180), fps=10.0)
    out = tmp_path / "out"
    results = showcase.main(["--controller", "scripted", "--episodes", "ep_049", "--max-steps", "40",
                             "--max-seconds", "2", "--panel", "96", "--grid-panel", "64", "--raw-dir", str(raw),
                             "--out-dir", str(out), "--skip-feasibility"])
    assert [r["id"] for r in results] == ["ep_049"] and results[0]["steps"] == 40
    for name in ("showcase_scripted.csv", "showcase_scripted_summary.txt", "showcase_scripted_overlay.png", "showcase_scripted_letters.png",
                 "showcase_scripted_ep_049.gif", "showcase_scripted.gif"):
        assert (out / name).stat().st_size > 0
    with open(out / "showcase_scripted.csv") as f:
        assert [row["id"] for row in csv.DictReader(f)] == ["ep_049"]


def test_sim_camera_has_x_up_and_y_left():
    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(split="showcase", episodes=["ep_049"])
    rec = showcase.SimRecorder(env, 32, showcase.VIEWS)
    env.reset(options={"episode_id": "ep_049"})
    o, _ = rec._project(np.array([[0.3, 0.0]]), 0.016)
    plus_x, _ = rec._project(np.array([[0.4, 0.0]]), 0.016)
    plus_y, _ = rec._project(np.array([[0.3, 0.1]]), 0.016)
    assert plus_x[0, 1] < o[0, 1] and abs(plus_x[0, 0] - o[0, 0]) < 1e-6       # +x: up the image
    assert plus_y[0, 0] < o[0, 0] and abs(plus_y[0, 1] - o[0, 1]) < 1e-6       # +y: left in the image
    env.close()
