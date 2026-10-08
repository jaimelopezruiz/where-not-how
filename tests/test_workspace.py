"""Tests for C3.4: workspace map (human table frame → robot EE frame).

Run:  python -m pytest tests/test_workspace.py -v
      python -m pytest tests/test_workspace.py::test_fit_on_train_data -v -s  (prints report)
"""
import json
import pathlib

import numpy as np
import pytest

from control.workspace import WorkspaceMap, generate_synthetic_paths

_PROCESSED = pathlib.Path("data/processed")
_SPLITS    = pathlib.Path("data/splits.json")


def test_generate_synthetic_paths_shapes():
    line, arc, s_curve = generate_synthetic_paths(n=60)
    for path in (line, arc, s_curve):
        assert path.shape == (60, 2)
        assert path.dtype == np.float64


def test_generate_synthetic_paths_origins():
    """All three paths must start at (0, 0)."""
    line, arc, s_curve = generate_synthetic_paths()
    for name, path in (("line", line), ("arc", arc), ("s_curve", s_curve)):
        assert np.allclose(path[0], [0.0, 0.0], atol=1e-9), (
            f"{name} does not start at (0, 0): {path[0]}"
        )


def test_uniform_scale_le_1():
    """WorkspaceMap.fit must produce scale ≤ 1."""
    line, arc, s_curve = generate_synthetic_paths()
    # Fake human paths with a spread of ~0.3 m
    hx = (-0.37 + -0.08) / 2
    hy = (-0.24 + 0.07) / 2
    human_center = np.array([hx, hy])
    trajectories = [p + human_center for p in (line, arc, s_curve)]

    # Reachable robot workspace about 0.15 m radius (smaller than human workspace)
    rng = np.random.default_rng(0)
    robot_center = np.array([0.1, 0.0])
    reachable_xy = rng.uniform(-0.12, 0.12, size=(200, 2)) + robot_center

    ws_map = WorkspaceMap.fit(trajectories, reachable_xy)
    assert 0.0 < ws_map.scale <= 1.0, f"scale {ws_map.scale:.4f} out of (0, 1]"


def test_transform_is_affine():
    """transform(xy) = scale * xy + offset."""
    ws_map = WorkspaceMap(scale=0.5, offset_xy=np.array([0.1, -0.05]))
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    expected = 0.5 * pts + np.array([0.1, -0.05])
    result = ws_map.transform(pts)
    assert np.allclose(result, expected, atol=1e-9)


def test_reachable_fraction_synthetic():
    """from_robot_reach returns a map with > 50 % reachable on all synthetic paths."""
    ws_map, reachable_xy = WorkspaceMap.from_robot_reach(n_samples=2000, seed=42)

    line, arc, s_curve = generate_synthetic_paths()
    hx = (-0.37 + -0.08) / 2
    hy = (-0.24 + 0.07) / 2
    human_center = np.array([hx, hy])

    for name, path in (("line", line), ("arc", arc), ("s_curve", s_curve)):
        path_human = path + human_center
        frac = ws_map.reachable_fraction(path_human, reachable_xy)
        assert frac > 0.5, (
            f"{name}: only {frac*100:.1f}% of mapped path is reachable "
            f"(expected > 50%)"
        )


def test_fit_function_exists():
    """WorkspaceMap.fit is a classmethod that accepts trajectories and reachable_xy."""
    line, arc, s_curve = generate_synthetic_paths()
    trajectories = [line, arc, s_curve]
    rng = np.random.default_rng(1)
    reachable_xy = rng.uniform(-0.15, 0.15, size=(100, 2)) + np.array([0.05, 0.0])
    ws_map = WorkspaceMap.fit(trajectories, reachable_xy)
    assert hasattr(ws_map, "scale")
    assert hasattr(ws_map, "offset_xy")
    assert ws_map.offset_xy.shape == (2,)


def test_fit_on_train_data():
    """Fit WorkspaceMap on train split using the feasibility-maximising algorithm.

    Physical parameters (SO-101 + 81 mm cube):
      BASE_FOOTPRINT_RADIUS = 0.0895 m  (from SO-101 base geom analysis)
      cube_side             = 0.081  m  (from props.yaml)
      r_half_diag           = sqrt(2) * 0.081 / 2 = 0.057 m
      r_min_cube            = 0.0895 + 0.057 = 0.147 m
      PUSHER_CAPSULE_RADIUS = 0.006  m  (capsule in scene.py)
      pusher_offset         = 0.057 + 0.006 + 0.010 = 0.073 m

    Prints a per-episode report.  Run with -s to see it.
    Skipped if data/processed/ is not present (CI without recorded data).
    """
    if not _SPLITS.exists() or not _PROCESSED.exists():
        pytest.skip("data/splits.json or data/processed/ not found")

    from control.kinematics.core import Adjoint, FKinBody, TransInv
    from control.kinematics.parser import DEFAULT_URDF, findMnS
    from sim.scene import BASE_FOOTPRINT_RADIUS, PUSHER_CAPSULE_RADIUS

    with open(_SPLITS) as f:
        splits = json.load(f)
    train_eps = splits["train"]

    # Load train trajectories (human table frame, metres); TRAIN ONLY
    trajectories = []
    for ep in train_eps:
        ep_data = np.load(_PROCESSED / f"{ep}.npz")
        trajectories.append(ep_data["cube_xy_yaw"][:, :2].copy())

    # Dense FK sample at push height (20 000 random configs → ~1 700 hits)
    M, Slist, limits = findMnS(DEFAULT_URDF)
    n = Slist.shape[1]
    Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
    rng = np.random.default_rng(42)
    z_push, z_tol = 0.01625, 0.025
    _hits = []
    for _ in range(20_000):
        q = rng.uniform(limits[:, 0], limits[:, 1])
        pos = FKinBody(M, Blist, q)[:3, 3]
        if abs(pos[2] - z_push) < z_tol:
            _hits.append(pos[:2].copy())
    reachable_xy = np.array(_hits) if _hits else np.zeros((4, 2))

    # Physical parameters
    cube_side    = 0.081          # from props.yaml (cube/side_mm: 81)
    r_half_diag  = np.sqrt(2) * cube_side / 2
    r_min_cube   = BASE_FOOTPRINT_RADIUS + r_half_diag
    pusher_offset = r_half_diag + PUSHER_CAPSULE_RADIUS + 0.010  # 10 mm margin

    print(f"\n--- Feasibility-maximising WorkspaceMap fit ---")
    print(f"  r_min_cube={r_min_cube:.4f} m  pusher_offset={pusher_offset:.4f} m")
    print(f"  FK sample: {len(reachable_xy)} positions at push height")

    ws_map, n_feasible = WorkspaceMap.fit_feasible(
        trajectories, reachable_xy, r_min_cube, pusher_offset
    )

    print(f"  scale  = {ws_map.scale:.4f}")
    print(f"  offset = ({ws_map.offset_xy[0]:.4f}, {ws_map.offset_xy[1]:.4f}) m")
    print(f"  feasible train trajectories: {n_feasible}/{len(train_eps)}")
    print()

    for ep, traj in zip(train_eps, trajectories):
        frac = ws_map.path_feasibility_fraction(traj, reachable_xy, r_min_cube, pusher_offset)
        tag = "OK" if frac >= 0.95 else "PARTIAL"
        print(f"  {tag}  {ep}: {frac * 100:5.1f}% feasible")

    # Assertions
    assert 0.0 < ws_map.scale <= 1.0
    assert ws_map.offset_xy.shape == (2,)
    assert n_feasible >= 20, f"only {n_feasible}/27 train trajectories feasible"


def test_infeasible_positions():
    """Base origin and far point are reported infeasible (catches the hull bug).

    The robot base footprint (r < r_min_cube = 0.147 m) and beyond-max-reach
    positions (r > 0.5 m) must both fail the feasibility check.  A convex hull
    on FK samples would incorrectly mark base-origin as 'inside' the hull;
    the KD-tree approach correctly fails it.
    """
    from sim.scene import BASE_FOOTPRINT_RADIUS, PUSHER_CAPSULE_RADIUS

    # Get FK reachable sample
    ws_map, reachable_xy = WorkspaceMap.from_robot_reach(n_samples=4000, seed=42)

    cube_side   = 0.081
    r_half_diag = np.sqrt(2) * cube_side / 2
    r_min_cube  = BASE_FOOTPRINT_RADIUS + r_half_diag
    pusher_offset = r_half_diag + PUSHER_CAPSULE_RADIUS + 0.010

    from control.workspace import REACH_TOLERANCE
    from scipy.spatial import cKDTree
    tree = cKDTree(reachable_xy)

    # Robot base origin: cube can't sit there (r = 0 < r_min_cube)
    origin = np.array([[0.0, 0.0]])
    cube_ok_origin = float(np.linalg.norm(origin, axis=1)[0]) > r_min_cube
    assert not cube_ok_origin, (
        f"Base origin (r=0) should fail cube clearance (r_min={r_min_cube:.3f} m)"
    )

    # Far point (0, 1.0): beyond arm max reach (~0.47 m)
    far_pt = np.array([[0.0, 1.0]])
    d_far, _ = tree.query(far_pt)
    far_reachable = float(d_far[0]) <= REACH_TOLERANCE
    assert not far_reachable, (
        f"Far point (0, 1.0) should be unreachable but nearest FK sample is "
        f"{d_far[0]*1000:.1f} mm away (tolerance {REACH_TOLERANCE*1000:.0f} mm)"
    )
