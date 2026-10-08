"""C4.3: smoke test of the whole chain on a synthetic clip, with no recorded data.

    python -m eval.smoke [--keep DIR]

A box is pushed along a known path in a rendered scene (tests/synth.py: the 2x2 board plus the cube marker,
measured sizes from data/props.yaml). The frames go through the extraction pipeline (marker detection, table
frame, cleaning, schema check), the resulting .npz is loaded into PushTrack-v0 as a single episode, the scripted
pusher runs it with evaluate()'s until-done protocol, and eval.metrics scores the cube path. It reads nothing
from data/raw/ and takes a few seconds; it also checks the extracted path against the rendered truth.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

from capture import live_check
from capture.common import load_props
from control.scripted_pusher import ScriptedPusher
from eval.report import evaluate
from extract import table as T
from extract.detect import RawEpisode, detect_frame
from extract.run import board_points, process
from tests import synth

SIZE = (1280, 720)
FPS = 15.0                  # rendering cost scales with frames; the cleaner works on the true timestamps
SPEED = 0.04                # m/s along the path, inside the recorded 3-9 cm/s
EPISODE = "smoke"
# Extracted path vs rendered truth. Detector corner scatter costs a few mm in the weak direction
# (tests/test_extract.py); the smoothed path of a real clip sits within ~2 mm.
EXTRACT_TOL_M = 0.006


def truth_table_path(speed=SPEED, fps=FPS):
    """Table-frame (t, x, y, yaw) of a gentle 14 cm curve inside the area the real recordings cover."""
    length = 0.14
    t = np.arange(int(length / speed * fps) + 1) / fps
    s = np.clip(speed * t / length, 0.0, 1.0)
    x = -0.30 + length * s
    y = -0.14 + 0.05 * s + 0.02 * np.sin(np.pi * s)
    yaw = 0.20 + 0.15 * s
    return t, np.column_stack([x, y, yaw])


def render_and_detect(t, table_xy_yaw, props):
    """Render each frame, detect markers, return (RawEpisode, K). The table frame is the board frame with y and yaw
    negated (OpenCV's board frame has y down and clockwise yaw), so the truth is converted on the way in."""
    K = synth.make_K(*SIZE)
    dist = np.zeros(5)
    detector = live_check.make_detector()
    board = live_check.make_board(props["board_marker_side"], props["board_gap"])
    T_cam_board = synth.look_at([-0.24, 0.52, -0.42], [-0.24, 0.12, 0.0], [0, 1, 0])
    n = len(t)
    nan44 = np.full((4, 4), np.nan)
    raw = RawEpisode(t=np.asarray(t, float), n_board=np.zeros(n, int), T_cam_board=np.full((n, 4, 4), np.nan),
                     board_rms=np.full(n, np.nan), board_px=np.full((n, 4, 2), np.nan),
                     T_cam_cube=np.full((n, 4, 4), np.nan), cube_rms=np.full(n, np.nan),
                     cube_corners=np.full((n, 4, 2), np.nan))
    for i, (x, y, yaw) in enumerate(table_xy_yaw):
        img, _ = synth.render_scene(K, dist, T_cam_board, SIZE, props["board_marker_side"], props["board_gap"],
                                    props["cube_marker_side"], (x, -y, -yaw), props["cube_height"], noise=1.0, seed=i)
        nb, Tb, brms, bpx, Tc, crms, ccor = detect_frame(img, detector, board, props, K, dist)
        raw.n_board[i], raw.board_rms[i], raw.board_px[i] = nb, brms, bpx
        raw.T_cam_board[i] = nan44 if Tb is None else Tb
        raw.T_cam_cube[i] = nan44 if Tc is None else Tc
        raw.cube_rms[i], raw.cube_corners[i] = crms, ccor
    return raw, K, dist


def extract_clip(raw, K, dist, props):
    """The extraction pipeline after detection, as extract.run does it for one episode."""
    track = T.board_track(raw)
    Tbc, pts = board_points(raw, track)
    # A 14 cm push cannot fix the plane's tilt across its own line (C2.1), which the real pipeline takes from other
    # episodes. The synthetic table is the board plane, so a zero-slope prior is exact here.
    plane = T.fit_plane(pts, prior=(0.0, 0.0))
    data, qa, t_off = process(EPISODE, "smoke", raw, track, Tbc, plane, K, dist, props)
    if data is None:
        raise RuntimeError(f"extraction failed: {qa['notes']}")
    return data, t_off


def run_smoke(workdir):
    """Run the chain; returns (metrics dict of the scripted pusher, extraction error in m, npz path)."""
    props = load_props()
    t_truth, truth = truth_table_path()
    raw, K, dist = render_and_detect(t_truth, truth, props)
    data, t_off = extract_clip(raw, K, dist, props)
    got = data["cube_xy_yaw"]
    want = np.column_stack([np.interp(data["t"] + t_off, t_truth, truth[:, i]) for i in range(2)])
    extract_err = float(np.abs(got[:, :2] - want).max())

    npz = Path(workdir) / f"{EPISODE}.npz"
    np.savez(npz, **data)

    from sim.push_env import PushTrackEnv
    env = PushTrackEnv(episode_files=[npz])
    try:
        result = evaluate(ScriptedPusher(), env=env, method="scripted")[0]
    finally:
        env.close()
    return result, extract_err, npz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", help="write the synthetic episode's .npz into this directory and keep it")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(args.keep) if args.keep else Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        result, extract_err, npz = run_smoke(workdir)
    print(f"extraction: cube path within {extract_err * 1000:.1f} mm of the rendered truth "
          f"(limit {EXTRACT_TOL_M * 1000:.0f} mm)")
    print(f"scripted pusher on the extracted episode ({result['steps']} steps):")
    for k in ("progress", "mean_deviation_cm", "final_error_cm", "success", "completion_time"):
        print(f"  {k:18s} {result[k]}")
    ok = bool(result["success"]) and extract_err < EXTRACT_TOL_M
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
