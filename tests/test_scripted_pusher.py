"""Tests for control.scripted_pusher (C5.1). No simulator: synthetic paths and a kinematic box.

Run: python -m pytest tests/test_scripted_pusher.py
"""
import numpy as np
import pytest

from control.scripted_pusher import (APPROACH, DONE, PUSH, ScriptedPusher, box_distance,
                                     contact_point, exit_distance)

SIDE = 0.081
H = SIDE / 2
R = 0.006
MARGIN = 0.005


def _pusher(**kw):
    return ScriptedPusher(cube_side=SIDE, pusher_radius=R, margin=MARGIN, **kw)


def _obs(ee, cube, yaw=0.0):
    return np.array([*ee, *cube, np.sin(yaw), np.cos(yaw)])


def _unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def _line(length=0.5, n=51, angle=0.0):
    s = np.linspace(0, length, n)
    return np.outer(s, _unit(angle))


def _arc(radius=0.25, sweep=np.pi / 2, n=80):
    th = np.linspace(0, sweep, n)
    return np.column_stack([radius * np.sin(th), radius * (1 - np.cos(th))])


# -- contact point and goal -----------------------------------------------------------------------

def test_goal_on_rear_face_axis_aligned():
    p = _pusher()
    p.reset(_line())
    p.act(_obs([-0.2, 0.0], [0.0, 0.0], 0.0))                       # d = +x
    assert p.d == pytest.approx([1.0, 0.0])
    assert p.goal == pytest.approx([-(H + R + MARGIN), 0.0])


@pytest.mark.parametrize("yaw", np.linspace(-np.pi, np.pi, 13))
@pytest.mark.parametrize("ang", np.linspace(0, 2 * np.pi, 9, endpoint=False))
def test_contact_on_boundary_and_correct_face(yaw, ang):
    d = _unit(ang)
    cube = np.array([0.1, -0.2])
    c = contact_point(cube, yaw, d, H)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    loc = rot.T @ (c - cube)
    assert np.max(np.abs(loc)) == pytest.approx(H, abs=1e-12)           # on the boundary
    assert box_distance(c, cube, yaw, H) == pytest.approx(0.0, abs=1e-12)
    # on the line through the centre, behind the box relative to d
    assert c - cube == pytest.approx(-exit_distance(d, yaw, H) * d)
    # the face hit is the one whose outward normal is most aligned with -d
    u = rot.T @ (-d)
    face = np.argmax(np.abs(u))
    assert abs(loc[face]) == pytest.approx(H, abs=1e-12)
    assert np.sign(loc[face]) == np.sign(u[face])


@pytest.mark.parametrize("yaw", [0.0, 0.3, np.pi / 4, 1.2, -0.7])
@pytest.mark.parametrize("ang", [0.0, 0.5, np.pi / 2, 2.0, -2.5])
def test_controller_goal_is_contact_pushed_out(yaw, ang):
    p = _pusher()
    d = _unit(ang)
    cube = np.array([0.2, 0.1])
    p.reset(np.array([cube, cube + 0.4 * d]))
    p.act(_obs(cube - 0.3 * d, cube, yaw))
    c = contact_point(cube, yaw, p.d, H)
    assert p.d == pytest.approx(d)
    assert p.goal == pytest.approx(c - (R + MARGIN) * d)
    assert np.linalg.norm(p.goal - cube) == pytest.approx(exit_distance(d, yaw, H) + R + MARGIN)


# -- approach never enters the inflated footprint -------------------------------------------------

def _drive(p, ee, cube, yaw, steps=400, stop_in_contact=True):
    """Move the pusher by its own actions around a fixed cube. Returns [(mode, ee), ...]."""
    ee = np.array(ee, float)
    log = []
    for _ in range(steps):
        a = p.act(_obs(ee, cube, yaw))
        log.append((p.mode, ee.copy()))
        if stop_in_contact and p.mode == PUSH and box_distance(ee, cube, yaw, H) <= R + 2 * p.push_speed:
            break
        ee = ee + a
    return log


STARTS = {
    "behind": [-0.30, 0.0], "front": [0.30, 0.0], "left": [0.0, 0.25], "right": [0.0, -0.25],
    "front_close": [0.075, 0.0], "behind_offset": [-0.15, 0.12], "diag": [0.12, 0.12],
    "on_ring": [0.0, 0.068], "touching_front": [H + R, 0.0], "touching_side": [0.0, H + R],
    "corner": [H + R, H + R],
}


@pytest.mark.parametrize("name", list(STARTS))
@pytest.mark.parametrize("yaw", [0.0, 0.4, np.pi / 4, -1.0])
def test_approach_stays_outside_footprint(name, yaw):
    p = _pusher()
    cube = np.array([0.3, 0.1])
    p.reset(np.array([cube, cube + [0.4, 0.0]]))
    log = _drive(p, cube + STARTS[name], cube, yaw)
    assert log[-1][0] == PUSH, "never lined up behind the box"
    for mode, ee in log[1:]:                                    # the start pose is given, not chosen
        sd = box_distance(ee, cube, yaw, H)
        # approach: clear of the footprint inflated by the pusher radius; pushing: within one push step of it
        floor = R + 0.5 * MARGIN if mode == APPROACH else R - p.push_speed
        assert sd >= floor - 1e-9, f"{name}: {mode} at distance {sd:.4f}"


def test_approach_from_front_goes_round_not_through():
    p = _pusher()
    cube = np.array([0.0, 0.0])
    p.reset(_line())
    log = _drive(p, [0.30, 0.0], cube, 0.0)
    path = np.array([e for _, e in log])
    assert path[:, 1].max() > H or path[:, 1].min() < -H        # swung out past a side of the box
    assert all(box_distance(e, cube, 0.0, H) > R for _, e in log[:-3])


def test_start_inside_ring_leaves_radially():
    p = _pusher()
    cube = np.array([0.0, 0.0])
    p.reset(_line())
    ee = np.array([H + R + 0.001, 0.0])                       # touching the front face
    a = p.act(_obs(ee, cube, 0.0))
    assert p.mode == APPROACH
    assert a[0] > 0 and abs(a[1]) < 1e-9


# -- actions --------------------------------------------------------------------------------------

@pytest.mark.parametrize("max_delta", [0.02, 0.005, 0.05])
def test_actions_respect_max_delta(max_delta):
    rng = np.random.default_rng(0)
    p = _pusher(max_delta=max_delta)
    p.reset(_arc())
    for _ in range(500):
        cube = rng.uniform(-0.1, 0.4, 2)
        ee = cube + rng.uniform(-0.4, 0.4, 2)
        a = p.act(_obs(ee, cube, rng.uniform(-np.pi, np.pi)))
        assert a.shape == (2,)
        assert np.all(np.abs(a) <= max_delta + 1e-12)
        assert np.hypot(*a) <= max_delta + 1e-12


def test_push_mode_goes_along_d_with_lateral_correction():
    p = _pusher()
    p.reset(_line())
    cube = np.array([0.1, 0.0])
    a = p.act(_obs([0.1 - H - R - 0.004, 0.0], cube))          # aligned, close to the face
    assert p.mode == PUSH
    assert a[0] == pytest.approx(p.push_speed) and a[1] == pytest.approx(0.0, abs=1e-12)
    a = p.act(_obs([0.1 - H - R - 0.004, 0.008], cube))        # 8 mm left of the push line: corrected to the right
    assert p.mode == PUSH and a[0] > 0 and a[1] < 0


def test_push_hysteresis():
    p = _pusher()
    p.reset(_line())
    cube = np.array([0.1, 0.0])
    x = 0.1 - H - R - 0.004
    p.act(_obs([x, 0.018], cube))                              # outside the entry band
    assert p.mode == APPROACH
    p.act(_obs([x, 0.010], cube))
    assert p.mode == PUSH
    p.act(_obs([x, 0.018], cube))                              # inside the exit band: keeps pushing
    assert p.mode == PUSH
    p.act(_obs([x, 0.030], cube))
    assert p.mode == APPROACH


# -- progress and lookahead -----------------------------------------------------------------------

def _walk(p, x, y=0.0, step=0.01):
    """Move the cube along +x from the path start to x in steps of at most 1 cm, acting at each one."""
    for cx in np.append(np.arange(0.0, x, step), x):
        p.act(_obs([cx - 0.2, y], [cx, y]))

def test_progress_is_monotone_and_never_jumps_back():
    p = _pusher()
    p.reset(_arc())
    ref = p.ref
    last = 0.0
    rng = np.random.default_rng(1)
    idx = np.r_[np.arange(len(ref)), np.arange(len(ref))[::-1], np.arange(len(ref))]   # forward, backward, forward
    for i in idx:
        cube = ref[i] + rng.normal(0, 0.003, 2)
        p.act(_obs(cube - [0.2, 0.0], cube))
        assert p.progress >= last
        last = p.progress
    assert last == pytest.approx(p.total, abs=0.01)


def test_progress_does_not_jump_across_self_crossing_path():
    # out along +x, up, back along -x just above, so the end of the path passes near the start
    ref = np.array([[0, 0], [0.4, 0], [0.4, 0.03], [0, 0.03], [0, 0.15]], float)
    p = _pusher(lookahead=0.03)
    p.reset(ref)
    p.act(_obs([-0.1, 0.0], [0.0, 0.0]))
    p.act(_obs([-0.1, 0.0], [0.1, 0.0]))
    # the cube sits closer to the return leg than to the outbound leg, but the search window is local
    p.act(_obs([-0.1, 0.0], [0.2, 0.0152]))
    assert p.progress < 0.3


def test_progress_starts_at_zero_and_ignores_cube_behind_start():
    p = _pusher()
    p.reset(_line())
    p.act(_obs([-0.3, 0.0], [-0.05, 0.0]))
    assert p.progress == 0.0
    assert p.lookahead_pt == pytest.approx([p.lookahead, 0.0])


def test_lookahead_is_fixed_arc_length_ahead():
    p = _pusher(lookahead=0.03)
    p.reset(_line(length=0.5))
    _walk(p, 0.2)
    assert p.progress == pytest.approx(0.2, abs=1e-9)
    assert p.lookahead_pt == pytest.approx([0.23, 0.0])


def test_lookahead_clamps_at_path_end():
    p = _pusher(lookahead=0.03)
    p.reset(_line(length=0.5))
    _walk(p, 0.49)
    assert p.lookahead_pt == pytest.approx([0.5, 0.0])
    assert p.d == pytest.approx([1.0, 0.0])
    assert p.mode != DONE                                       # 1 cm from the end is not done yet (stop_tol 5 mm)


def test_done_at_path_end_gives_zero_action():
    p = _pusher()
    p.reset(_line(length=0.5))
    _walk(p, 0.495)
    a = p.act(_obs([0.4, 0.0], [0.498, 0.0]))
    assert p.mode == DONE
    assert np.all(a == 0.0)


def test_cube_past_end_is_pushed_back():
    p = _pusher()
    p.reset(_line(length=0.5))
    _walk(p, 0.5)
    p.act(_obs([0.4, 0.0], [0.56, 0.0]))                        # overshot by 6 cm
    assert p.mode != DONE
    assert p.d == pytest.approx([-1.0, 0.0])


def test_reset_clears_state_and_rejects_degenerate_paths():
    p = _pusher()
    p.reset(_line())
    _walk(p, 0.3)
    assert p.progress > 0.25
    p.reset(_line(length=0.2))
    assert p.progress == 0.0 and p.mode == APPROACH
    with pytest.raises(ValueError):
        p.reset([[0.1, 0.1], [0.1, 0.1]])


def test_cube_side_comes_from_props_not_a_default():
    p = ScriptedPusher()
    assert p.half_side == pytest.approx(0.0405)


# -- closed loop on a kinematic box (translation only: the box is pushed out of the capsule) -------

def _push_out(ee, cube, yaw):
    """Minimal translation of the cube that clears the capsule (a frictionless point-contact model)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    q = rot.T @ (ee - cube)
    near = np.clip(q, -H, H)
    v = q - near
    dist = np.hypot(*v)
    if dist >= R:
        return cube
    if dist < 1e-12:                                            # centre inside the box: shove along the nearest face normal
        i = int(np.argmin(H - np.abs(q)))
        shove = np.zeros(2)
        shove[i] = np.sign(q[i]) * (H - abs(q[i]) + R)
        return cube - rot @ shove
    return cube - rot @ (v / dist) * (R - dist)


def _closed_loop(ref, yaw, ee0, steps=1500):
    p = _pusher()
    p.reset(ref)
    cube, ee = ref[0].copy(), np.array(ee0, float)
    trace, bad = [cube.copy()], 0
    for _ in range(steps):
        a = p.act(_obs(ee, cube, yaw))
        if p.mode == DONE:
            break
        ee = ee + a
        cube = _push_out(ee, cube, yaw)
        bad += box_distance(ee, cube, yaw, H) < R - 0.011       # pusher well inside the box
        trace.append(cube.copy())
    return p, np.array(trace), bad


@pytest.mark.parametrize("yaw", [0.0, 0.3])
def test_closed_loop_straight_path(yaw):
    ref = _line(0.3)
    p, trace, bad = _closed_loop(ref, yaw, [-0.1, 0.1])
    assert p.mode == DONE
    assert np.linalg.norm(trace[-1] - ref[-1]) < 0.01
    assert bad == 0


@pytest.mark.parametrize("yaw", [0.0, 0.5])
def test_closed_loop_curved_path(yaw):
    from eval.metrics import compute_metrics
    ref = _arc(radius=0.3, sweep=np.pi / 2)
    p, trace, bad = _closed_loop(ref, yaw, [-0.12, 0.05])
    m = compute_metrics(ref, trace)
    assert m["success"], m
    assert m["mean_deviation_cm"] < 1.5, m
    assert bad == 0
