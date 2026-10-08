"""Tests for scripts.results_figures (C11.3), on small synthetic CSVs.

Run:  python -m pytest tests/test_results_figures.py
"""
import csv
import math

import numpy as np
import pytest

from scripts import results_figures as rf

FIELDS = ["id", "category", "method", "progress", "mean_deviation_cm", "final_error_cm", "success", "completion_time"]


def _write_results(path, method, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (cat, ok, dev, fin, t) in enumerate(rows):
            w.writerow({"id": f"ep_{i:03d}", "category": cat, "method": method, "progress": 1.0 if ok else 0.4,
                        "mean_deviation_cm": dev, "final_error_cm": fin, "success": ok, "completion_time": t})


def test_results_table_overall_and_by_category(tmp_path):
    a, b = tmp_path / "a_test.csv", tmp_path / "b_test.csv"
    _write_results(a, "scripted", [("straight", True, 0.4, 0.3, 4.0), ("straight", True, 0.6, 0.5, 6.0),
                                   ("turn", True, 1.0, 0.4, 8.0)])
    _write_results(b, "ppo", [("straight", True, 2.0, 1.0, 10.0), ("straight", False, 5.0, 12.0, "nan"),
                              ("turn", False, 6.0, 20.0, "")])
    text = rf.results_table([a, b])
    lines = text.splitlines()
    assert "| scripted | 3 | 3/3 (100%) | 0.67 | 0.40 | 6.0 |" in lines      # means over the three episodes
    assert "| ppo | 3 | 1/3 (33%) | 4.33 | 11.00 | 10.0 |" in lines            # completion time: successes only
    assert "| scripted | straight | 2 | 2/2 (100%) | 0.50 | 0.40 | 5.0 |" in lines
    assert "| ppo | turn | 1 | 0/1 (0%) | 6.00 | 20.00 | n/a |" in lines
    assert text.index("scripted") < text.index("ppo")                           # argument order kept


def test_main_writes_the_table_and_prints_it(tmp_path, capsys):
    a = tmp_path / "scripted_test.csv"
    _write_results(a, "", [("curve", True, 0.8, 0.4, 4.5)])
    rf.main([str(a), "--out-dir", str(tmp_path / "out")])
    written = (tmp_path / "out" / "results_table.md").read_text()
    assert "| scripted | 1 | 1/1 (100%) |" in written and written in capsys.readouterr().out   # name from the file stem
    assert b"\r" not in (tmp_path / "out" / "results_table.md").read_bytes()


def test_the_real_scripted_test_csv_gives_the_known_numbers():
    from capture.common import REPO_ROOT
    text = rf.results_table([REPO_ROOT / "results" / "scripted_test.csv"])
    assert "| scripted | 7 | 7/7 (100%) | 0.61 | 0.39 |" in text


def _write_curve(path, header, rows, ragged=False):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
        if ragged:
            f.write("999\n")                          # a half-written last line, as in a run in progress


def test_read_curve_concatenates_files_sorts_by_timesteps_and_keeps_the_later_duplicate(tmp_path):
    cols = ["time/total_timesteps", "rollout/success_rate", "other"]
    _write_curve(tmp_path / "progress_0-3M.csv", cols, [[4096, "", 1], [8192, 0.1, 1], [12288, 0.2, 1]])
    _write_curve(tmp_path / "progress.csv", cols, [[16384, 0.4, 1], [12288, 0.3, 1], [20480, 0.5, 1]], ragged=True)
    files = rf.run_files(tmp_path, None, "progress*.csv")
    assert [f.name for f in files] == ["progress_0-3M.csv", "progress.csv"]      # by first timestep, not by name
    t, y = rf.read_curve(files, "time/total_timesteps", ["rollout/success_rate"])["rollout/success_rate"]
    assert list(t) == [8192, 12288, 16384, 20480]                     # the empty-value row and the ragged row are dropped
    assert dict(zip(t, y))[12288] == 0.3                              # the later file wins at a repeated timestep


def test_moving_average_keeps_length_and_smooths():
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = rf.moving_average(y, 3)
    assert len(out) == len(y) and out[2] == pytest.approx(2 / 3)
    assert np.array_equal(rf.moving_average(y, 1), y)


def test_curves_figure_is_written_for_two_runs_with_several_progress_files(tmp_path):
    for name, scale in (("plain", 0.0), ("res", 1.0)):
        d = tmp_path / name
        d.mkdir()
        _write_curve(d / "eval_curve.csv", ["timesteps", "wall_s", "success_rate", "mean_deviation_cm"],
                     [[k * 1000, 1, scale * k / 5, 3.0 - k * 0.1] for k in range(1, 6)])
        cols = ["time/total_timesteps", "rollout/success_rate"]
        _write_curve(d / "progress_0-3M.csv", cols, [[k * 512, scale * k / 40] for k in range(1, 6)])
        _write_curve(d / "progress.csv", cols, [[k * 512, scale * k / 40] for k in range(6, 11)])
    out = tmp_path / "out"
    rf.main(["--run", f"plain PPO={tmp_path / 'plain'}", "--run", f"residual={tmp_path / 'res'}",
             "--out-dir", str(out), "--smooth", "3"])
    assert (out / "training_curves.png").stat().st_size > 5000


def test_a_run_without_curves_is_an_error_naming_the_pattern(tmp_path):
    with pytest.raises(SystemExit, match="eval_curve"):
        rf.main(["--run", f"x={tmp_path}", "--out-dir", str(tmp_path)])
