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


# ── evaluate stub ─────────────────────────────────────────────────────────────

def test_evaluate_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate(lambda obs: np.zeros(2), "test")
