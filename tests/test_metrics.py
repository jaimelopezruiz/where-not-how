"""Tests for eval.metrics (C4.1).

All paths are synthetic; no real episode data is loaded.
Run: python -m pytest tests/test_metrics.py
"""
import math

import numpy as np
import pytest

from eval.metrics import compute_metrics, category_summary, SUCCESS_FINAL_M, SUCCESS_PROGRESS


# ── Synthetic path builders ───────────────────────────────────────────────────

def _line(n=50, length=1.0):
    """Straight reference from (0, 0) to (length, 0), n points."""
    return np.column_stack([np.linspace(0, length, n), np.zeros(n)])


def _arc(n=60, radius=0.5, angle=math.pi / 2):
    """Quarter-arc reference (90 deg)."""
    theta = np.linspace(0, angle, n)
    return np.column_stack([radius * np.sin(theta), radius * (1 - np.cos(theta))])


def _s_curve(n=100):
    """S-curve reference (one full period of a sine wave along x)."""
    x = np.linspace(0, 1.0, n)
    y = 0.05 * np.sin(2 * math.pi * x)
    return np.column_stack([x, y])


# ── Perfect tracking ──────────────────────────────────────────────────────────

def test_perfect_tracking_line():
    ref = _line(50, 1.0)
    m = compute_metrics(ref, ref.copy())
    assert m["progress"] == pytest.approx(1.0, abs=1e-9)
    assert m["mean_deviation_cm"] == pytest.approx(0.0, abs=1e-6)
    assert m["final_error_cm"] == pytest.approx(0.0, abs=1e-6)
    assert m["success"] is True


def test_perfect_tracking_arc():
    ref = _arc()
    m = compute_metrics(ref, ref.copy())
    assert m["progress"] == pytest.approx(1.0, abs=1e-6)
    assert m["mean_deviation_cm"] < 1e-4
    assert m["success"] is True


def test_perfect_tracking_s_curve():
    ref = _s_curve()
    m = compute_metrics(ref, ref.copy())
    assert m["progress"] == pytest.approx(1.0, abs=1e-6)
    assert m["mean_deviation_cm"] < 1e-4
    assert m["success"] is True


# ── Constant lateral offset gives exact mean deviation ───────────────────────

def test_lateral_offset_1cm_on_line():
    """Achieved path = reference shifted 1 cm in y -> mean_deviation = 1 cm."""
    ref = _line(50, 1.0)
    achieved = ref.copy()
    achieved[:, 1] += 0.01          # 1 cm lateral shift

    m = compute_metrics(ref, achieved)
    # Every point is 1 cm from the line.
    assert m["mean_deviation_cm"] == pytest.approx(1.0, abs=1e-4)
    # The last achieved point (1, 0.01) projects to (1, 0): arc = 1.0.
    assert m["progress"] == pytest.approx(1.0, abs=1e-6)


def test_lateral_offset_3cm():
    ref = _line(50, 1.0)
    achieved = ref.copy()
    achieved[:, 1] += 0.03
    m = compute_metrics(ref, achieved)
    assert m["mean_deviation_cm"] == pytest.approx(3.0, abs=1e-4)


# ── Success threshold: final error ───────────────────────────────────────────
# Reference: 10 cm straight line [(0,0) -> (0.1, 0)]

REF_10CM = np.array([[0.0, 0.0], [0.1, 0.0]])


def test_success_final_error_just_inside_2cm():
    """final_error = 1.9 cm, progress = 1.0  ->  success = True."""
    achieved = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.019]])
    m = compute_metrics(REF_10CM, achieved)
    assert m["final_error_cm"] == pytest.approx(1.9, abs=0.01)
    assert m["progress"] == pytest.approx(1.0, abs=1e-6)
    assert m["success"] is True


def test_success_final_error_just_outside_2cm():
    """final_error = 2.1 cm, progress = 1.0  ->  success = False."""
    achieved = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.021]])
    m = compute_metrics(REF_10CM, achieved)
    assert m["final_error_cm"] == pytest.approx(2.1, abs=0.01)
    assert m["progress"] == pytest.approx(1.0, abs=1e-6)
    assert m["success"] is False


def test_success_final_error_exactly_2cm():
    """final_error = 2.0 cm  ->  success = False (strict <)."""
    achieved = np.array([[0.0, 0.0], [0.05, 0.0], [0.1, 0.02]])
    m = compute_metrics(REF_10CM, achieved)
    assert m["final_error_cm"] == pytest.approx(2.0, abs=0.01)
    assert m["success"] is False


# ── Success threshold: progress ───────────────────────────────────────────────
# Use a 4 cm reference so an 89%/91% endpoint can be close (< 2 cm) to the end.

REF_4CM = np.array([[0.0, 0.0], [0.04, 0.0]])   # total_len = 0.04 m


def test_success_progress_just_below_90pct():
    """progress ≈ 0.89 < 0.9, final_error < 2 cm  ->  success = False."""
    # Last point projects to t = 0.89 along the segment; ends 0.5 cm off the line.
    achieved = np.array([[0.0, 0.0], [0.02, 0.0], [0.0356, 0.005]])
    m = compute_metrics(REF_4CM, achieved)
    assert m["progress"] < SUCCESS_PROGRESS
    assert m["final_error_cm"] < SUCCESS_FINAL_M * 100.0   # final_error is fine
    assert m["success"] is False


def test_success_progress_just_above_90pct():
    """progress ≈ 0.975, final_error < 2 cm  ->  success = True."""
    achieved = np.array([[0.0, 0.0], [0.02, 0.0], [0.039, 0.005]])
    m = compute_metrics(REF_4CM, achieved)
    assert m["progress"] >= SUCCESS_PROGRESS
    assert m["final_error_cm"] < SUCCESS_FINAL_M * 100.0
    assert m["success"] is True


# ── Path that stalls halfway ──────────────────────────────────────────────────

def test_stalls_halfway_progress():
    """Achieved goes from 0 to 0.5 m then stops: progress ≈ 0.5."""
    ref = _line(50, 1.0)
    half = np.column_stack([np.linspace(0, 0.5, 25), np.zeros(25)])
    # Pad with the stalled position.
    stalled = np.vstack([half, np.tile(half[-1], (10, 1))])
    m = compute_metrics(ref, stalled)
    assert m["progress"] == pytest.approx(0.5, abs=1e-6)
    assert m["success"] is False


def test_stalls_at_30pct():
    ref = _line(50, 1.0)
    partial = np.column_stack([np.linspace(0, 0.3, 30), np.zeros(30)])
    m = compute_metrics(ref, partial)
    assert m["progress"] == pytest.approx(0.3, abs=1e-6)


# ── Completion time ───────────────────────────────────────────────────────────

def test_completion_time_perfect_tracking():
    """Perfect tracking: completion_time is finite and > 0."""
    ref = _line(50, 0.1)       # 10 cm, so final error = 0 once we reach the end
    t = np.linspace(0.0, 5.0, 50)
    m = compute_metrics(ref, ref.copy(), t=t)
    assert not math.isnan(m["completion_time"])
    assert m["completion_time"] >= 0.0
    assert m["completion_time"] <= 5.0


def test_completion_time_nan_when_stalled():
    """Path that never reaches the end: completion_time = nan."""
    ref = _line(50, 1.0)
    half = np.column_stack([np.linspace(0, 0.5, 50), np.zeros(50)])
    t = np.linspace(0.0, 5.0, 50)
    m = compute_metrics(ref, half, t=t)
    assert m["success"] is False
    assert math.isnan(m["completion_time"])


def test_completion_time_none_without_timestamps():
    """Without t, completion_time is always nan."""
    ref = _line(50, 0.1)
    m = compute_metrics(ref, ref.copy())   # no t
    assert math.isnan(m["completion_time"])


def test_completion_time_earlier_on_faster_path():
    """A path that finishes in 2 s should have an earlier completion time than one in 4 s."""
    ref = _line(50, 0.1)
    t_fast = np.linspace(0.0, 2.0, 50)
    t_slow = np.linspace(0.0, 4.0, 50)
    m_fast = compute_metrics(ref, ref.copy(), t=t_fast)
    m_slow = compute_metrics(ref, ref.copy(), t=t_slow)
    assert m_fast["completion_time"] < m_slow["completion_time"]


# ── Category summary ──────────────────────────────────────────────────────────

def test_category_summary_basic():
    results = [
        {"category": "straight", "progress": 1.0, "mean_deviation_cm": 0.5,
         "final_error_cm": 1.0, "success": True, "completion_time": 3.2},
        {"category": "straight", "progress": 0.8, "mean_deviation_cm": 2.0,
         "final_error_cm": 3.0, "success": False, "completion_time": float("nan")},
        {"category": "curve", "progress": 0.95, "mean_deviation_cm": 1.2,
         "final_error_cm": 1.5, "success": True, "completion_time": 4.0},
    ]
    s = category_summary(results)
    assert set(s.keys()) == {"straight", "curve"}
    assert s["straight"]["n"] == 2
    assert s["straight"]["success_rate"] == pytest.approx(0.5)
    assert s["straight"]["mean_progress"] == pytest.approx(0.9)
    assert s["straight"]["mean_deviation_cm"] == pytest.approx(1.25)
    # Only the successful straight episode has a finite completion time.
    assert s["straight"]["mean_completion_time"] == pytest.approx(3.2)
    assert s["curve"]["success_rate"] == pytest.approx(1.0)
    assert s["curve"]["n"] == 1


def test_category_summary_all_failed_time_is_nan():
    results = [
        {"category": "turn", "progress": 0.5, "mean_deviation_cm": 5.0,
         "final_error_cm": 10.0, "success": False, "completion_time": float("nan")},
    ]
    s = category_summary(results)
    assert math.isnan(s["turn"]["mean_completion_time"])
    assert s["turn"]["success_rate"] == pytest.approx(0.0)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_single_segment_reference():
    """A reference with just 2 vertices (one segment) works correctly."""
    ref = np.array([[0.0, 0.0], [1.0, 0.0]])
    m = compute_metrics(ref, ref.copy())
    assert m["progress"] == pytest.approx(1.0)
    assert m["mean_deviation_cm"] == pytest.approx(0.0, abs=1e-9)
    assert m["success"] is True


def test_progress_cannot_exceed_1():
    """Achieved path that overshoots the end: progress is capped at 1.0."""
    ref = _line(10, 0.5)
    # Achieved overshoots by 20 cm beyond the end.
    overshoot = np.column_stack([np.linspace(0, 0.7, 20), np.zeros(20)])
    m = compute_metrics(ref, overshoot)
    assert m["progress"] <= 1.0


def test_arc_path_mean_deviation_exact():
    """Points 2 cm outside an arc have mean deviation ≈ 2 cm."""
    ref = _arc(60)
    # Scale each reference point outward from the arc's centre by 0.02 m.
    # Centre of the quarter-arc is at (0, 0.5) (radius 0.5).
    centre = np.array([0.0, 0.5])
    vecs = ref - centre
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    achieved = ref + (vecs / norms) * 0.02
    m = compute_metrics(ref, achieved)
    assert m["mean_deviation_cm"] == pytest.approx(2.0, abs=0.02)


# ── Arc-length weighting: stationary prefix ───────────────────────────────────

def test_stationary_prefix_does_not_change_mean_deviation():
    """Prepending a long stationary prefix must not lower the mean deviation."""
    ref = _line(50, 1.0)

    # Base path: 1 cm lateral offset, 30 uniformly spaced moving points.
    x = np.linspace(0, 1.0, 30)
    base = np.column_stack([x, np.ones(30) * 0.01])

    # Path with a 100-sample stationary prefix at the start position.
    prefix = np.tile(base[0], (100, 1))
    with_prefix = np.vstack([prefix, base])

    m_base = compute_metrics(ref, base)
    m_prefix = compute_metrics(ref, with_prefix)

    # The stationary prefix has ds = 0 so it contributes nothing; result must match.
    assert m_prefix["mean_deviation_cm"] == pytest.approx(m_base["mean_deviation_cm"], abs=1e-9)


def test_stationary_only_path_reports_start_deviation():
    """A completely stationary achieved path reports the start-point deviation."""
    ref = _line(20, 1.0)
    # Cube sits at (0.5, 0.03) for all samples: 3 cm from the line.
    static = np.tile([0.5, 0.03], (50, 1))
    m = compute_metrics(ref, static)
    assert m["mean_deviation_cm"] == pytest.approx(3.0, abs=1e-4)
