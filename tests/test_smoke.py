"""C4.3: synthetic clip -> extraction -> PushTrack-v0 -> scripted pusher -> metrics, in one test.

Run from the repo root:  python -m pytest tests/test_smoke.py   (or: python -m eval.smoke)
"""
import numpy as np

from eval import smoke


def test_synthetic_clip_runs_through_extraction_env_pusher_and_metrics(tmp_path):
    result, extract_err, npz = smoke.run_smoke(tmp_path)

    d = np.load(npz)                                         # what extraction wrote is what the env was given
    assert set(d.files) >= {"t", "cube_xy_yaw", "finger_xy", "contact", "cube_marker_side"}
    assert d["contact"].any() and np.hypot(*np.diff(d["cube_xy_yaw"][[0, -1], :2], axis=0)[0]) > 0.10
    assert extract_err < smoke.EXTRACT_TOL_M                 # extracted path matches the rendered truth

    assert result["id"] == smoke.EPISODE
    assert result["success"]                                 # final error < 2 cm and progress >= 90 %
    assert result["progress"] >= 0.9 and result["final_error_cm"] < 2.0
    assert result["mean_deviation_cm"] < 2.0
