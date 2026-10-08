"""Tests for eval.report (C4.2 env-independent parts).

Run: python -m pytest tests/test_report.py
"""
import csv
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from eval.report import write_results_csv, category_table, plot_overlay, write_gif, evaluate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_results(n=4):
    rows = [
        {"id": "ep_000", "category": "straight", "method": "scripted",
         "progress": 1.0, "mean_deviation_cm": 0.8, "final_error_cm": 0.5,
         "success": True, "completion_time": 3.1},
        {"id": "ep_001", "category": "straight", "method": "scripted",
         "progress": 0.7, "mean_deviation_cm": 4.0, "final_error_cm": 8.0,
         "success": False, "completion_time": float("nan")},
        {"id": "ep_002", "category": "curve", "method": "scripted",
         "progress": 0.95, "mean_deviation_cm": 1.5, "final_error_cm": 1.2,
         "success": True, "completion_time": 4.5},
        {"id": "ep_003", "category": "turn", "method": "scripted",
         "progress": 0.85, "mean_deviation_cm": 2.0, "final_error_cm": 2.5,
         "success": False, "completion_time": float("nan")},
    ]
    return rows[:n]


def _dummy_frame(h=16, w=24, fill=128):
    return np.full((h, w, 3), fill, dtype=np.uint8)


# ── write_results_csv ─────────────────────────────────────────────────────────

def test_write_results_csv_creates_file():
    results = _fake_results()
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "results.csv"
        write_results_csv(results, p)
        assert p.exists()


def test_write_results_csv_columns():
    results = _fake_results(2)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        write_results_csv(results, p)
        with open(p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert set(rows[0].keys()) >= {"id", "category", "method", "progress",
                                        "mean_deviation_cm", "final_error_cm",
                                        "success", "completion_time"}


def test_write_results_csv_values():
    results = _fake_results(1)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.csv"
        write_results_csv(results, p)
        with open(p) as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["id"] == "ep_000"
        assert rows[0]["category"] == "straight"
        assert rows[0]["success"] == "True"


def test_write_results_csv_creates_parent_dirs():
    results = _fake_results(1)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "sub" / "dir" / "results.csv"
        write_results_csv(results, p)
        assert p.exists()


# ── category_table ────────────────────────────────────────────────────────────

def test_category_table_contains_categories():
    table = category_table(_fake_results())
    assert "straight" in table
    assert "curve" in table
    assert "turn" in table


def test_category_table_is_string():
    table = category_table(_fake_results())
    assert isinstance(table, str)
    assert len(table) > 0


# ── plot_overlay ──────────────────────────────────────────────────────────────

def _ep(cat="straight", n=20, success=True):
    x = np.linspace(0, 0.5, n)
    return {
        "ref_xy": np.column_stack([x, np.zeros(n)]),
        "achieved_xy": np.column_stack([x, np.ones(n) * 0.005]),
        "category": cat,
        "success": success,
    }


def test_plot_overlay_creates_file():
    eps = [_ep("straight"), _ep("curve", success=False)]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "overlay.png"
        plot_overlay(eps, p)
        assert p.exists()
        assert p.stat().st_size > 0


def test_plot_overlay_creates_parent_dirs():
    eps = [_ep()]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nested" / "overlay.png"
        plot_overlay(eps, p)
        assert p.exists()


# ── write_gif ─────────────────────────────────────────────────────────────────

def test_write_gif_creates_file():
    real = [_dummy_frame(16, 24, fill=100) for _ in range(5)]
    sim = [_dummy_frame(16, 24, fill=200) for _ in range(5)]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.gif"
        write_gif(real, sim, p, fps=5.0)
        assert p.exists()
        assert p.stat().st_size > 0


def test_write_gif_pads_height():
    """Frames with different heights should not crash."""
    real = [_dummy_frame(16, 24) for _ in range(3)]
    sim = [_dummy_frame(32, 24) for _ in range(3)]   # taller
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.gif"
        write_gif(real, sim, p, fps=5.0)
        assert p.exists()


def test_write_gif_length_mismatch_raises():
    real = [_dummy_frame() for _ in range(3)]
    sim = [_dummy_frame() for _ in range(4)]
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError, match="mismatch"):
            write_gif(real, sim, Path(d) / "out.gif")


def test_write_gif_empty_raises():
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            write_gif([], [], Path(d) / "out.gif")


def test_write_gif_greyscale_frames():
    """Greyscale frames (H, W) should be accepted and converted to RGB."""
    real = [np.full((16, 24), 100, dtype=np.uint8) for _ in range(3)]
    sim = [np.full((16, 24), 200, dtype=np.uint8) for _ in range(3)]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.gif"
        write_gif(real, sim, p, fps=5.0)
        assert p.exists()


# ── evaluate ──────────────────────────────────────────────────────────────────

class _StubEnv:
    """PushTrack-like env on explicit paths: the cube follows the last action exactly (no physics)."""
    def __init__(self, paths, ids=None, terminate_at=None):
        self._paths, self.episode_ids, self.dt, self.cube = paths, ids, 0.05, None
        self.terminate_at = terminate_at

    def reset(self, options=None):
        self._i = options["episode_id"] and self.episode_ids.index(options["episode_id"])             if "episode_id" in options else options["episode"]
        self.path, self.k = self._paths[self._i], 0
        return np.r_[self.path[0], self.path[0], 0.0, 1.0], {}

    @property
    def reference_path(self):
        return self.path.copy()

    @property
    def cube_xy_yaw(self):
        return self.path[self.k].copy(), 0.0

    def step(self, action):
        self.k = min(self.k + 1, len(self.path) - 1)
        last = len(self.path) - 1
        done = self.k == last if self.terminate_at is None else self.k >= self.terminate_at
        return np.r_[self.path[self.k], self.path[self.k], 0.0, 1.0], 0.0, done, self.k == last, {}


class _FollowRef:
    def reset(self, ref_xy):
        self.ref = ref_xy

    def __call__(self, obs):
        return np.zeros(2)


def test_evaluate_scores_stub_env_and_resets_controller():
    paths = [np.column_stack([np.linspace(0, 0.3, 40), np.zeros(40)]),
             np.column_stack([np.linspace(0, 0.2, 30), np.linspace(0, 0.1, 30)])]
    env = _StubEnv(paths, ids=["ep_000", "ep_005"])
    ctrl = _FollowRef()
    res = evaluate(ctrl, env=env, method="stub")
    assert [r["id"] for r in res] == ["ep_000", "ep_005"]
    assert all(r["method"] == "stub" and r["success"] for r in res)
    assert all(r["mean_deviation_cm"] == pytest.approx(0.0, abs=1e-9) for r in res)
    assert res[0]["category"] == "straight"                       # from data/manifest.csv
    assert res[0]["steps"] == 39 and res[0]["completion_time"] == pytest.approx(37 * 0.05)   # first step within 2 cm of the end (0.3 m path, 39 steps)
    assert ctrl.ref == pytest.approx(paths[1])                    # reset() got the last episode's reference
    with tempfile.TemporaryDirectory() as d:
        write_results_csv(res, Path(d) / "r.csv")
        plot_overlay(res, Path(d) / "o.png")
        header = (Path(d) / "r.csv").read_text().splitlines()[0]
        assert header.startswith("id,category,method,progress")
        assert (Path(d) / "o.png").exists()


def test_evaluate_real_env_on_synthetic_path():
    """Plumbing through PushTrackEnv: reset by path index, step, metrics. Cube never touched, so it fails."""
    from sim.push_env import PushTrackEnv
    path = np.column_stack([np.linspace(0.30, 0.40, 20), np.full(20, 0.10)])
    env = PushTrackEnv(paths=[path], max_episode_steps=3)
    res = evaluate(lambda obs: np.zeros(2), env=env, method="idle")
    env.close()
    assert len(res) == 1 and res[0]["id"] == "path_0" and res[0]["steps"] == 3
    assert res[0]["achieved_xy"].shape == (4, 2)
    assert res[0]["progress"] == pytest.approx(0.0, abs=1e-3) and not res[0]["success"]
    assert res[0]["achieved_xy"][0] == pytest.approx(res[0]["ref_xy"][0], abs=1e-4)   # cube starts on the path


def test_evaluate_needs_split_or_env():
    with pytest.raises(ValueError):
        evaluate(lambda obs: np.zeros(2))


class _DoneAfter:
    """Controller that reports done once it has been asked for n actions."""
    def __init__(self, n):
        self.n, self.calls = n, 0

    def __call__(self, obs):
        self.calls += 1
        return np.zeros(2)

    @property
    def done(self):
        return self.calls >= self.n


def test_until_done_runs_past_the_envs_first_success():
    path = np.column_stack([np.linspace(0, 0.3, 40), np.zeros(40)])
    env = _StubEnv([path], ids=["ep_000"], terminate_at=5)          # env reports success at step 5
    run_on = evaluate(_DoneAfter(20), env=env)[0]
    assert run_on["steps"] == 19 and len(run_on["achieved_xy"]) == 20     # stops when the controller says done
    first = evaluate(_DoneAfter(20), env=env, until_done=False)[0]
    assert first["steps"] == 5                                            # training behaviour: stop at success


def test_controller_without_done_still_stops_at_env_termination():
    path = np.column_stack([np.linspace(0, 0.3, 40), np.zeros(40)])
    env = _StubEnv([path], ids=["ep_000"], terminate_at=5)
    assert evaluate(lambda obs: np.zeros(2), env=env)[0]["steps"] == 5


def test_until_done_stops_at_the_step_limit():
    path = np.column_stack([np.linspace(0, 0.3, 40), np.zeros(40)])
    env = _StubEnv([path], ids=["ep_000"], terminate_at=5)
    assert evaluate(_DoneAfter(10_000), env=env)[0]["steps"] == 39         # env truncates at the end of the path
