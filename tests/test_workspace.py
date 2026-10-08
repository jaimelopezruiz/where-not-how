"""Tests for C3.4: workspace map (human table frame → robot EE frame).

Run:  python -m pytest tests/test_workspace.py -v
"""
import numpy as np
import pytest

from control.workspace import WorkspaceMap, generate_synthetic_paths


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
