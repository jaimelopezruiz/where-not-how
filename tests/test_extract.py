"""extract/: table frame, cube pose, cleaning, schema and split (T2).

Run from the repo root:  python -m pytest tests/test_extract.py   (or: python -m tests.test_extract)
"""
import json

import numpy as np
import pytest

from capture import live_check
from capture.common import REPO_ROOT
from extract import clean as C
from extract import schema, split
from extract import table as T
from extract.detect import RawEpisode, detect_frame
from tests import synth

PROPS = {"board_marker_side": 0.0564, "board_gap": 0.0376, "cube_marker_side": 0.047}
H = 0.0325                      # marker height above the table


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def T4(R, p):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, p
    return out


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# ---------------------------------------------------------------- table frame and cube pose, pose level

def truth_path(n=120):
    """A curved path with a yaw sweep through +-pi, in the table frame."""
    s = np.linspace(0, 1, n)
    return np.column_stack([-0.30 + 0.18 * s, -0.20 + 0.25 * s + 0.05 * np.sin(5 * s), np.pi * (2 * s - 1) * 1.1])


def fake_episode(path, r_tilt, hold=(40, 90), hz=30.0):
    """RawEpisode built analytically: the table plane tilted from the board plane by ``r_tilt``, a moving camera.

    Table axes in board coordinates are the columns of ``r_tilt @ diag(1, -1, -1)``: untilted, x is the board's x,
    y its flipped y, z up (board z points into the table). A tilt about the board's y or x axis keeps the board's x
    axis in the table plane, which is what the table frame's x direction is defined to be, so the truth is exact
    and written independently of ``Plane.axes``. The camera drifts linearly (so interpolation across the hold is
    exact) with random shake on the frames outside the hold, which the per-frame board pose has to absorb.
    """
    n = len(path)
    t = np.arange(n) / hz
    axes = r_tilt @ np.diag([1.0, -1.0, -1.0])                  # columns: table x, y, z (up) in board coordinates
    n_up = axes[:, 2]
    T_board_marker = np.stack([T4(axes @ Rz(yaw), H * n_up + x * axes[:, 0] + y * axes[:, 1]) for x, y, yaw in path])
    rng = np.random.default_rng(3)
    shake = rng.normal(0, 1, (n, 6)) * [0.002, 0.002, 0.002, 0.001, 0.001, 0.003]
    shake[max(hold[0] - 3, 0):hold[1] + 3] = 0
    T_cam_board = np.stack([T4(Rx(np.radians(130)) @ Rz(0.02 * i / n + shake[i, 0]) @ Ry(0.01 * i / n + shake[i, 1]) @ Rx(shake[i, 2]),
                               [0.1 + 0.05 * i / n + shake[i, 3], 0.05 + shake[i, 4], 0.6 + 0.02 * i / n + shake[i, 5]])
                            for i in range(n)])
    n_board = np.full(n, 4)
    n_board[hold[0]:hold[1]] = 2                                # arm over the board: 2-marker poses are not used
    junk = T_cam_board.copy()
    junk[hold[0]:hold[1], 2, 3] += 0.02                         # the 10-25 mm depth wander of a 2-marker pose
    return RawEpisode(t=t, n_board=n_board, T_cam_board=junk, board_rms=np.full(n, 0.75),
                      board_px=np.full((n, 4, 2), np.nan), T_cam_cube=T_cam_board @ T_board_marker,
                      cube_rms=np.full(n, 0.2), cube_corners=np.full((n, 4, 2), np.nan))


@pytest.mark.parametrize("r_tilt", [Ry(np.radians(4.5)), Ry(np.radians(-7)), Rx(np.radians(5)), np.eye(3)])
def test_cube_pose_in_table_frame_matches_truth_with_tilt_shake_and_hold(r_tilt):
    """Tilted table, moving camera, a 50-frame board occlusion with garbage 2-marker poses: exact table-frame pose."""
    path = truth_path()
    raw = fake_episode(path, r_tilt)
    track = T.board_track(raw)
    assert track.measured.sum() == len(raw) - 50                   # 2-marker frames are never used
    assert track.longest_hold_s == pytest.approx(51 / 30)
    Tbc = T.cube_in_board(track.T_cam_board, raw.T_cam_cube)
    plane = T.fit_plane(Tbc[track.measured][:, :3, 3])
    x, y, yaw, flat, z = T.cube_to_table(plane, Tbc)
    assert np.abs(x - path[:, 0]).max() < 1e-6                     # metres, held frames included
    assert np.abs(y - path[:, 1]).max() < 1e-6
    assert np.abs(wrap(yaw - path[:, 2])).max() < 1e-6
    assert plane.tilt_deg == pytest.approx(np.degrees(np.arccos(r_tilt[2, 2])), abs=1e-6)
    assert plane.height == pytest.approx(H, abs=1e-9)
    assert plane.resid_rms < 1e-9
    assert np.degrees(flat.max()) < 1e-3                           # marker flat on the table (arccos near 1 is float-noisy)


def test_table_frame_is_z_up_and_yaw_counter_clockwise():
    """+x along the board's x, +y is the board's y flipped, a positive yaw turns x towards y (seen from above)."""
    plane = T.Plane(a=-H, b=0.0, c=0.0, resid_rms=0, resid_max=0, spread=1, prior_weight=0)
    A = plane.axes()
    assert np.allclose(A, np.diag([1.0, -1.0, -1.0]))             # board y down -> table y flipped, board z into table -> z up
    assert np.linalg.det(A) == pytest.approx(1.0)
    # OpenCV board frame (clockwise yaw): marker x axis at +30 deg about board z must read as -30 deg here
    Tm = T4(Rz(np.radians(30)) @ np.diag([1, -1, -1]), [0.2, 0.1, -H])
    x, y, yaw, tilt, z = T.cube_to_table(plane, Tm[None])
    assert (x[0], y[0]) == pytest.approx((0.2, -0.1))
    assert yaw[0] == pytest.approx(np.radians(-30))
    assert z[0] == pytest.approx(0, abs=1e-12) and tilt[0] == pytest.approx(0, abs=1e-9)


def test_plane_prior_fixes_the_tilt_a_straight_push_cannot():
    """Points along one line leave the cross-slope open; the prior supplies it, and well-spread points ignore it."""
    rng = np.random.default_rng(1)
    b, c = -0.08, 0.01
    s = np.linspace(0, 0.3, 80)
    line = np.column_stack([s, 0.02 * s, -H + b * s + c * 0.02 * s]) + rng.normal(0, 0.0005, (80, 3)) * [0, 0, 1]
    free = T.fit_plane(line)
    assert free.spread < 1e-3 and free.prior_weight == 0
    anchored = T.fit_plane(line, prior=(b, c))
    assert anchored.prior_weight > 0.9
    assert np.allclose(anchored.slope, (b, c), atol=0.01)
    xy = rng.uniform(0, 0.3, (200, 2))
    spread = np.column_stack([xy, -H + b * xy[:, 0] + c * xy[:, 1]])
    fitted = T.fit_plane(spread, prior=(0.5, 0.5))                   # a bad prior barely matters
    assert np.allclose(fitted.slope, (b, c), atol=0.02)
    assert T.prior_slope([free, fitted]) == pytest.approx([b, c], abs=0.02)   # only the spread one counts


def test_fill_board_pose_interpolates_and_holds():
    t = np.arange(10, dtype=float)
    Ts = np.stack([T4(Rz(0.1 * i), [i, 0, 0.5]) for i in range(10)])
    measured = np.ones(10, bool)
    measured[[0, 4, 5, 6, 9]] = False
    Ts_bad = Ts.copy()
    Ts_bad[~measured] = np.nan
    out = T.fill_board_pose(t, Ts_bad, measured)
    want = Ts.copy()
    want[0], want[9] = Ts[1], Ts[8]                                      # the ends are held, not extrapolated
    assert np.allclose(out, want, atol=1e-12)                            # linear motion: interpolation is exact
    with pytest.raises(ValueError):
        T.fill_board_pose(t, Ts_bad, np.eye(10, dtype=bool)[0])


def test_board_track_rejects_high_reprojection_error():
    raw = fake_episode(truth_path(), np.eye(3), hold=(0, 0))
    raw.board_rms[10:13] = 3.0
    assert T.board_track(raw).measured.sum() == len(raw) - 3


def test_pixel_drift_and_hold_deviation():
    n = 50
    t = np.arange(n) / 30
    px = np.tile(np.array([[100.0, 100], [200, 100], [100, 200], [200, 200]]), (n, 1, 1))
    px[:, :, 0] += np.linspace(0, 3, n)[:, None]                           # camera slides 3 px in x
    measured = np.ones(n, bool)
    measured[20:30] = False
    px[20:30] = np.nan
    px[22:25, 0] = px[0, 0] + [3, 0] + np.linspace(0, 3, n)[22:25, None] * 0   # one marker seen 3 px off the held track
    first_last, mx = T.board_pixel_drift(px, measured)
    assert first_last == pytest.approx(3.0) and mx == pytest.approx(3.0)
    assert T.board_hold_deviation(px, measured, t) > 1.0


def test_recovered_marker_side_matches_geometry():
    """Corner rays against the plane give back the marker's own size, independent of the pose solver."""
    K = synth.make_K()
    T_cam_board = synth.look_at([0.0, 0.5, -0.45], [-0.15, 0.1, 0.0], [0, 1, 0])
    plane = T.Plane(a=-H, b=0.0, c=0.0, resid_rms=0, resid_max=0, spread=1, prior_weight=0)
    side = 0.047
    h = side / 2
    local = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    Tm = T4(Rz(0.4) @ np.diag([1, -1, -1]), [-0.2, 0.15, -H])
    cam = ((T_cam_board @ Tm)[:3, :3] @ local.T).T + (T_cam_board @ Tm)[:3, 3]
    pix = (K @ cam.T).T
    corners = (pix[:, :2] / pix[:, 2:3])[None]
    got = T.recovered_marker_side(plane, T_cam_board[None], corners, K, np.zeros(5))
    assert got == pytest.approx(side, abs=1e-6)


# ---------------------------------------------------------------- the whole front end on rendered images

def test_rendered_scene_end_to_end_matches_table_frame_truth():
    """Rendered board + cube, detected and converted: catches OpenCV's y-down / z-in frame or a yaw sign leaking through."""
    size = (1280, 720)
    K = synth.make_K(*size)
    dist = np.zeros(5)
    detector, board = live_check.make_detector(), live_check.make_board(PROPS["board_marker_side"], PROPS["board_gap"])
    T_cam_board = synth.look_at([-0.10, 0.50, -0.42], [-0.10, 0.08, 0.0], [0, 1, 0])
    truth_board = [(-0.25, 0.10, np.radians(10)), (-0.20, 0.14, np.radians(35)), (-0.15, 0.18, np.radians(-20))]
    rows = []
    for x, y, yaw in truth_board:                     # cube pose in the OpenCV board frame (y down, yaw clockwise)
        img, _ = synth.render_scene(K, dist, T_cam_board, size, PROPS["board_marker_side"], PROPS["board_gap"],
                                    PROPS["cube_marker_side"], (x, y, yaw), H)
        r = detect_frame(img, detector, board, PROPS, K, dist)
        assert r[0] == 4 and r[1] is not None and r[4] is not None
        rows.append(r)
    Tcb = np.stack([r[1] for r in rows])
    Tcc = np.stack([r[4] for r in rows])
    plane = T.Plane(a=-H, b=0.0, c=0.0, resid_rms=0, resid_max=0, spread=1, prior_weight=0)
    x, y, yaw, tilt, z = T.cube_to_table(plane, T.cube_in_board(Tcb, Tcc))
    want = np.array(truth_board)
    # detector corner scatter at this view costs up to ~4 mm in the weak direction (tests/test_live_check.py)
    assert np.abs(x - want[:, 0]).max() < 0.005
    assert np.abs(y + want[:, 1]).max() < 0.005                    # table y is board y flipped
    assert np.abs(wrap(yaw + want[:, 2])).max() < np.radians(1.0)  # table yaw is board yaw negated


# ---------------------------------------------------------------- cleaning

def test_gaps_up_to_five_frames_are_interpolated_longer_are_left():
    n = 60
    t = np.arange(n) / 30
    v = np.column_stack([t * 0.05, t * 0.02, np.zeros(n)])
    valid = np.ones(n, bool)
    valid[10:15] = False           # 5 frames: bridged
    valid[30:36] = False           # 6 frames: left
    valid[:3] = False              # leading run: never filled
    out, ok = C.interpolate_gaps(t, v, valid)
    assert ok[10:15].all() and np.allclose(out[10:15], v[10:15])
    assert not ok[30:36].any() and np.isnan(out[30:36]).all()
    assert not ok[:3].any()
    assert C.longest_gap_between(valid) == 6


def test_clean_trajectory_leaves_long_gaps_nan_and_never_smooths_across_them():
    n = 120
    t = np.arange(n) / 30
    v = np.column_stack([0.05 * t, np.where(t < 2, 0.0, 0.3), np.zeros(n)])      # y steps 0 -> 0.3 inside the gap
    keep = np.ones(n, bool)
    keep[55:75] = False
    cl = C.clean_trajectory(t, v, keep)
    assert cl.max_gap == 20
    gap = np.isnan(cl.xy_yaw[:, 0])
    assert gap.any() and cl.n_interpolated == 0
    before = cl.xy_yaw[(cl.t + cl.t_offset) < t[55] - 1e-9][:, 1]
    after = cl.xy_yaw[(cl.t + cl.t_offset) > t[75] + 1e-9][:, 1]
    assert np.nanmax(np.abs(before)) < 1e-9 and np.allclose(after[np.isfinite(after)], 0.3)   # no smearing of the step


def test_yaw_is_unwrapped_before_filtering():
    n = 90
    t = np.arange(n) / 30
    true = np.linspace(2.5, 4.5, n)                                        # crosses +pi
    v = np.column_stack([np.zeros(n), np.zeros(n), wrap(true)])
    cl = C.clean_trajectory(t, v, np.ones(n, bool))
    grid = cl.t + cl.t_offset
    assert np.abs(np.diff(cl.xy_yaw[:, 2])).max() < 0.2                    # no 2*pi jump left
    assert np.abs(cl.xy_yaw[:, 2] - np.interp(grid, t, true)).max() < 1e-6   # a ramp is untouched by SG, on the original branch


def test_resample_uses_true_timestamps_and_control_rate():
    n = 90
    t = np.cumsum(np.random.default_rng(0).uniform(0.025, 0.04, n))          # jittery camera clock
    v = np.column_stack([0.1 * t, -0.05 * t, 0.3 * t])
    cl = C.clean_trajectory(t, v, np.ones(n, bool))
    assert np.allclose(np.diff(cl.t), 1 / C.CONTROL_HZ)
    grid = cl.t + cl.t_offset
    assert np.abs(cl.xy_yaw - np.column_stack([0.1 * grid, -0.05 * grid, 0.3 * grid])).max() < 1e-6   # linear motion survives SG


def test_savgol_on_times_equals_scipy_on_uniform_times_and_is_exact_on_jittery_ones():
    from scipy.signal import savgol_filter
    rng = np.random.default_rng(2)
    t = np.arange(80) / 30
    v = np.column_stack([np.cumsum(rng.normal(0, 1, 80)), rng.normal(0, 1, 80)])
    assert np.allclose(C.savgol_times(t, v, 11, 2), savgol_filter(v, 11, 2, axis=0, mode="interp"), atol=1e-9)
    tj = t + rng.uniform(-0.012, 0.012, 80)
    quad = np.column_stack([1 + 2 * tj + 3 * tj ** 2, -tj])                       # a quadratic passes through unchanged
    assert np.allclose(C.savgol_times(tj, quad, 11, 2), quad, atol=1e-9)


def test_hampel_drops_a_single_frame_pose_flip():
    n = 60
    v = np.column_stack([np.linspace(0, 0.1, n), np.zeros(n), np.zeros(n)])
    v[30, 2] = np.radians(80)                                               # one IPPE-style flip
    v[40, 0] += 0.05
    ok = C.confidence_mask(np.ones(n, bool), np.full(n, 0.2), np.zeros(n), v)
    assert not ok[30] and not ok[40] and ok.sum() == n - 2


def test_confidence_mask_drops_high_error_and_tilted_frames():
    n = 30
    v = np.column_stack([np.linspace(0, 0.05, n), np.zeros(n), np.zeros(n)])
    rms = np.full(n, 0.2)
    rms[5] = 3.0
    tilt = np.zeros(n)
    tilt[9] = np.radians(40)
    det = np.ones(n, bool)
    det[12] = False
    ok = C.confidence_mask(det, rms, tilt, np.where(det[:, None], v, np.nan))
    assert not ok[[5, 9, 12]].any() and ok.sum() == n - 3


def test_contact_window_follows_speed_and_does_not_assume_a_still_start():
    t = np.arange(200) / C.CONTROL_HZ
    speed = np.where((t > 2) & (t < 6), 0.04, 0.0)
    x = np.cumsum(speed / C.CONTROL_HZ)
    xy = np.column_stack([x, np.zeros_like(x)])
    contact = C.contact_mask(t, xy)
    assert contact[(t > 2.3) & (t < 5.7)].all() and not contact[t < 1.5].any() and not contact[t > 6.5].any()
    moving_at_start = np.column_stack([0.04 * t, np.zeros_like(t)])           # box already moving in frame 0
    c2 = C.contact_mask(t, moving_at_start)
    assert c2[0] and c2.all()
    paused = np.column_stack([np.cumsum(np.where((t > 4) & (t < 4.3), 0.0, 0.04) / C.CONTROL_HZ), np.zeros_like(t)])
    assert C.contact_mask(t, paused).all()                                     # a 0.3 s pause stays inside the push


# ---------------------------------------------------------------- schema

def valid_episode(m=60):
    t = np.arange(m) / C.CONTROL_HZ
    return {"t": t, "cube_xy_yaw": np.column_stack([-0.3 + 0.2 * t / t[-1], -0.1 + 0 * t, 0 * t]),
            "finger_xy": np.full((m, 2), np.nan), "contact": np.ones(m, bool), "cube_marker_side": np.float64(0.0472)}


def test_schema_accepts_a_valid_episode():
    assert schema.problems(valid_episode(), PROPS) == []


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.pop("contact"), "missing keys"),
    (lambda d: d.update(cube_xy_yaw=d["cube_xy_yaw"][:, :2]), "cube_xy_yaw shape"),
    (lambda d: d.update(cube_xy_yaw=d["cube_xy_yaw"] * [1000, 1000, 1]), "not in metres"),
    (lambda d: d.update(cube_xy_yaw=d["cube_xy_yaw"] * [100, 100, 1]), "not in metres"),
    (lambda d: d["cube_xy_yaw"].__setitem__((5, 0), np.nan), "NaN"),
    (lambda d: d["t"].__setitem__(7, d["t"][6]), "strictly increasing"),
    (lambda d: d.update(t=d["t"] * 1.5), "uniform"),
    (lambda d: d.update(contact=d["contact"].astype(int)), "contact"),
    (lambda d: d.update(contact=np.zeros(60, bool)), "no contact"),
    (lambda d: d.update(cube_marker_side=np.float64(0.0520)), "recovered marker side"),
    (lambda d: d.update(cube_xy_yaw=np.zeros((60, 3))), "less than 5 cm"),
    (lambda d: d["cube_xy_yaw"].__setitem__((slice(30, None), 2), 7.0), "unwrapped"),
    (lambda d: d.update(t=d["t"].astype(np.float32)), "float64"),
])
def test_schema_flags_each_defect(mutate, fragment):
    d = valid_episode()
    mutate(d)
    errs = schema.problems(d, PROPS)
    assert any(fragment in e for e in errs), errs


def processed_files():
    return sorted((REPO_ROOT / "data" / "processed").glob("ep_*.npz"))


def test_every_real_episode_passes_the_schema():
    files = processed_files()
    if not files:
        pytest.skip("no processed episodes yet (python -m extract.run)")
    for f in files:
        assert schema.validate_file(f) == [], f.name


# ---------------------------------------------------------------- fingertip

def test_tip_on_plane_recovers_table_points_at_the_tip_height():
    """Project known table-frame points at z_tip through a tilted-table scene and cut them back with the same plane."""
    from extract import fingertip as F
    K = synth.make_K()
    r_tilt = Ry(np.radians(4.5))
    axes = r_tilt @ np.diag([1.0, -1.0, -1.0])
    up = axes[:, 2]
    # z = a + b x + c y has up = (b, c, -1)/norm and sits H above the board origin along up
    plane = T.Plane(a=H / up[2], b=-up[0] / up[2], c=-up[1] / up[2], resid_rms=0, resid_max=0, spread=1, prior_weight=0)
    assert np.allclose(plane.normal_up, up, atol=1e-12) and plane.height == pytest.approx(H)
    T_cam_board = np.tile(synth.look_at([-0.1, 0.5, -0.42], [-0.1, 0.08, 0.0], [0, 1, 0]), (4, 1, 1))
    z_tip = 0.008 - H
    truth = np.array([[-0.25, -0.10, z_tip], [-0.2, 0.0, z_tip], [-0.15, -0.2, z_tip], [-0.3, 0.05, z_tip]])
    P_board = H * up + truth[:, 0:1] * axes[:, 0] + truth[:, 1:2] * axes[:, 1] + truth[:, 2:3] * up
    cam = (T_cam_board[:, :3, :3] @ P_board[:, :, None])[:, :, 0] + T_cam_board[:, :3, 3]
    pix = (K @ cam.T).T
    pix = pix[:, :2] / pix[:, 2:3]
    pix[3] = np.nan
    got = F.tip_on_plane(pix, T_cam_board, plane, z_tip, K, np.zeros(5))
    assert np.abs(got[:3] - truth[:3]).max() < 1e-9
    assert np.isnan(got[3]).all()


def test_finger_props_are_never_defaulted(tmp_path):
    from extract import fingertip as F
    p = tmp_path / "props.yaml"
    p.write_text("cube:\n  height_mm: 32.5\n")
    with pytest.raises(SystemExit, match="finger.tip_radius_mm"):
        F.load_finger_props(p)
    p.write_text("cube:\n  height_mm: 32.5\nfinger:\n  tip_radius_mm: 9\n")
    assert F.load_finger_props(p) == pytest.approx((0.009, 0.0325))


def test_onto_grid_bridges_short_gaps_only():
    from extract import fingertip as F
    t = np.arange(60) / 30
    xy = np.column_stack([t, 2 * t])
    xy[10:14] = np.nan                       # 4 frames: bridged
    xy[30:40] = np.nan                       # 10 frames: left
    grid = np.arange(0, 1.9, 0.05)
    out = F.onto_grid(t, xy, grid)
    assert np.isfinite(out[(grid > 0.3) & (grid < 0.4)]).all() and np.allclose(out[:, 1][np.isfinite(out[:, 1])], 2 * grid[np.isfinite(out[:, 1])])
    assert np.isnan(out[(grid > 1.05) & (grid < 1.25)]).all()


# ---------------------------------------------------------------- split

EPS = ([(f"ep_{i:03d}", "straight") for i in range(10)] + [(f"ep_{i:03d}", "curve") for i in range(10, 21)] +
       [(f"ep_{i:03d}", "turn") for i in range(21, 29)] + [(f"ep_{i:03d}", "multi") for i in range(29, 34)] +
       [(f"ep_{i:03d}", "showcase") for i in range(34, 38)])


def test_split_is_deterministic_disjoint_stratified_and_excludes_showcase():
    a, b = split.make_split(EPS, seed=0), split.make_split(list(reversed(EPS)), seed=0)
    assert a == b                                                              # manifest order is irrelevant
    assert a != split.make_split(EPS, seed=1)
    assert not set(a["train"]) & set(a["test"])
    assert set(a["train"]) | set(a["test"]) == {e for e, c in EPS if c != "showcase"}
    assert a["showcase"] == [e for e, c in EPS if c == "showcase"]
    assert not set(a["showcase"]) & (set(a["train"]) | set(a["test"]))
    assert a["per_category"] == {"curve": {"train": 9, "test": 2}, "multi": {"train": 4, "test": 1},
                                 "straight": {"train": 8, "test": 2}, "turn": {"train": 6, "test": 2}}


def test_manifest_reader_skips_discards_and_short_rows(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("id,category,notes\nep_000,straight,\nep_001,discard\nep_002,Curve,a note\nep_003,discard,x\n")
    assert split.read_manifest(m) == [("ep_000", "straight"), ("ep_002", "curve")]


def test_frozen_split_matches_the_manifest_and_processed_data():
    path = REPO_ROOT / "data" / "splits.json"
    if not path.exists():
        pytest.skip("split not written yet (python -m extract.split)")
    frozen = json.loads(path.read_text())
    assert frozen == split.make_split(split.read_manifest(), frozen["seed"], frozen["test_fraction"])
    have = {f.stem for f in processed_files()}
    if have:
        assert set(frozen["train"]) | set(frozen["test"]) | set(frozen["showcase"]) == have


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
