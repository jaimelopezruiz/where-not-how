"""Extraction pipeline (T2): raw video -> data/processed/ep_XXX.npz, results/qa_extraction.csv, overlay plot.

    python -m extract.run                       # every manifest row whose category is not "discard"
    python -m extract.run --episodes ep_000     # a subset (the plane prior then comes from that subset only)
    python -m extract.run --cache-dir DIR       # reuse/store per-episode detections (development; delete to redo)

Stages: detect (extract.detect) -> board pose per frame and table plane (extract.table) -> cleaning
(extract.clean) -> schema check (extract.schema). Time comes from the per-frame CSV timestamps.
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from capture.common import REPO_ROOT, load_intrinsics, load_props
from extract import clean as C
from extract import table as T
from extract.detect import RawEpisode, detect_episode, episode_paths
from extract.plot import plot_overlay
from extract.schema import problems
from extract.split import read_manifest

RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"
QA_CSV = REPO_ROOT / "results" / "qa_extraction.csv"
OVERLAY = REPO_ROOT / "results" / "trajectories_overlay.png"

# Warning thresholds for the QA notes column
DRIFT_WARN_PX = 1.0         # static camera: 0.0-0.3 px
HOLD_DEV_WARN_PX = 2.0
TILT_DEV_WARN_DEG = 3.0
SIDE_WARN = 0.03

QA_COLUMNS = [
    "episode", "category", "status", "frames", "duration_s", "frame_dt_max_ms",
    "cube_detected_frac", "dropped_frames", "interpolated_frames", "max_gap_frames",
    "board_4marker_frac", "board_longest_hold_s", "board_drift_px", "board_drift_max_px", "board_hold_dev_p95_px",
    "plane_spread_mm", "plane_prior_weight", "plane_tilt_deg", "plane_resid_rms_mm", "plane_resid_max_mm",
    "marker_side_mm", "marker_side_err_pct", "marker_tilt_max_deg", "smooth_resid_xy_mm",
    "start_speed_mm_s", "contact_frac", "path_length_cm", "yaw_range_deg", "n_samples", "notes",
]


def _detect_one(args):
    ep, cache_dir = args
    cache = Path(cache_dir) / f"{ep}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        return ep, RawEpisode.load(cache)
    props = load_props()
    K, dist, size = load_intrinsics()
    video, csv_path = episode_paths(RAW_DIR, ep)
    raw = detect_episode(video, csv_path, props, K, dist, size)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.save(cache)
    return ep, raw


def detect_all(eps, cache_dir=None, workers=8):
    with ProcessPoolExecutor(min(workers, len(eps))) as ex:
        return dict(ex.map(_detect_one, [(e, cache_dir) for e in eps]))


def board_points(raw, track):
    """Cube marker positions in the board frame, for frames where the board pose was measured, not filled."""
    Tbc = T.cube_in_board(track.T_cam_board, raw.T_cam_cube)
    use = track.measured & np.isfinite(Tbc[:, 0, 0])
    return Tbc, Tbc[use][:, :3, 3]


def process(ep, category, raw, track, Tbc, plane, K, dist, props):
    """One episode after detection: returns (npz dict or None, QA row, camera-clock time of t == 0)."""
    qa = dict.fromkeys(QA_COLUMNS, "")
    qa.update(episode=ep, category=category, frames=len(raw), duration_s=round(float(raw.t[-1]), 2),
              frame_dt_max_ms=round(float(np.diff(raw.t).max() * 1000), 1))
    notes = []
    x, y, yaw, tilt, _ = T.cube_to_table(plane, Tbc)
    values = np.column_stack([x, y, yaw])
    detected = np.isfinite(x)
    keep = C.confidence_mask(detected, raw.cube_rms, tilt, values)
    d_first_last, d_max = T.board_pixel_drift(raw.board_px, track.measured)
    side = T.recovered_marker_side(plane, track.T_cam_board, raw.cube_corners, K, dist)
    hold_dev = T.board_hold_deviation(raw.board_px, track.measured, raw.t)
    qa.update(cube_detected_frac=round(float(detected.mean()), 4),
              board_4marker_frac=round(float(track.measured.mean()), 3),
              board_longest_hold_s=round(track.longest_hold_s, 2),
              board_drift_px=round(d_first_last, 2), board_drift_max_px=round(d_max, 2),
              board_hold_dev_p95_px=round(hold_dev, 2),
              plane_spread_mm=round(plane.spread * 1000, 1), plane_prior_weight=round(plane.prior_weight, 2),
              plane_tilt_deg=round(plane.tilt_deg, 2), plane_resid_rms_mm=round(plane.resid_rms * 1000, 2),
              plane_resid_max_mm=round(plane.resid_max * 1000, 1), marker_side_mm=round(side * 1000, 2),
              marker_side_err_pct=round((side / props["cube_marker_side"] - 1) * 100, 2),
              marker_tilt_max_deg=round(float(np.degrees(np.nanmax(tilt))), 1))
    if d_first_last > DRIFT_WARN_PX:
        notes.append(f"camera moved {d_first_last:.1f} px (per-frame board pose absorbs it)")
    if hold_dev > HOLD_DEV_WARN_PX:
        notes.append(f"partly visible board markers {hold_dev:.1f} px off the held track (p95)")
    if abs(side / props["cube_marker_side"] - 1) > SIDE_WARN:
        notes.append("recovered marker size off by more than 3%")
    if detected.mean() < 1:
        notes.append(f"cube marker missing in {int((~detected).sum())} frames")
    if not keep.any():
        qa.update(status="error", notes="no usable cube frames")
        return None, qa, 0.0
    cl = C.clean_trajectory(raw.t, values, keep)
    pose = cl.xy_yaw
    xy = pose[:, :2]
    ok = np.isfinite(xy[:, 0])
    v = np.hypot(*np.gradient(xy[ok], cl.t[ok], axis=0).T) if ok.sum() > 2 else np.zeros(1)
    qa.update(dropped_frames=cl.n_dropped, interpolated_frames=cl.n_interpolated, max_gap_frames=cl.max_gap,
              smooth_resid_xy_mm=round(float(np.hypot(*cl.raw_resid[:2]) * 1000), 2),
              start_speed_mm_s=round(float(v[0] * 1000), 1), contact_frac=round(float(cl.contact.mean()), 3),
              path_length_cm=round(float(np.hypot(*np.diff(xy[ok], axis=0).T).sum() * 100), 1),
              yaw_range_deg=round(float(np.degrees(np.ptp(pose[ok, 2]))), 1), n_samples=len(cl.t))
    if cl.max_gap > C.MAX_GAP:
        notes.append(f"gap of {cl.max_gap} frames left unbridged")
    data = {"t": cl.t, "cube_xy_yaw": pose, "finger_xy": np.full((len(cl.t), 2), np.nan),
            "contact": cl.contact, "cube_marker_side": np.float64(side)}
    errs = problems(data, props)
    qa.update(status="ok" if not errs else "error")
    notes += errs
    qa["notes"] = "; ".join(notes)
    return (data if not errs else None), qa, cl.t_offset


def write_qa(rows, path=QA_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, QA_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def run(episodes, cache_dir=None, out_dir=OUT_DIR, qa_path=QA_CSV, overlay_path=OVERLAY, workers=8, finger=None):
    props = load_props()
    K, dist, _ = load_intrinsics()
    cats = dict(episodes)
    eps = sorted(cats)
    raws = detect_all(eps, cache_dir, workers)
    tracks = {e: T.board_track(raws[e]) for e in eps}
    pts = {e: board_points(raws[e], tracks[e]) for e in eps}
    prior = T.prior_slope([T.fit_plane(pts[e][1]) for e in eps])
    if prior is None:
        print("warning: no episode spreads enough to fit the plane alone; fitting each without a prior")
    rows, plots = [], []
    for e in eps:
        plane = T.fit_plane(pts[e][1], prior)
        data, qa, t_off = process(e, cats[e], raws[e], tracks[e], pts[e][0], plane, K, dist, props)
        if data is not None:
            if finger is not None:
                data["finger_xy"] = finger(e, raws[e], tracks[e], plane, data["t"] + t_off)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f"{e}.npz", **data)
            plots.append((e, cats[e], data["cube_xy_yaw"], data["contact"]))
        rows.append(qa)
        print(f"{e} {cats[e]:9s} {qa['status']:5s} {qa['notes']}")
    write_qa(rows, qa_path)
    if plots:
        plot_overlay(plots, overlay_path)
    return rows, prior


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", nargs="*", help="episode ids (default: all non-discard manifest rows)")
    ap.add_argument("--cache-dir", help="store/reuse per-episode detections here (development only)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--finger", action="store_true",
                    help="also track the fingertip (C2.3): needs the hand model and finger.tip_radius_mm in props.yaml")
    args = ap.parse_args(argv)
    manifest = read_manifest()
    if args.episodes:
        manifest = [m for m in manifest if m[0] in args.episodes]
        if len(manifest) != len(args.episodes):
            raise SystemExit("some --episodes are discarded or not in the manifest")
    finger = None
    if args.finger:
        from extract.fingertip import FingerTracker
        finger = FingerTracker(*load_intrinsics()[:2])
    rows, _ = run(manifest, args.cache_dir, workers=args.workers, finger=finger)
    if finger is not None:
        print("hand found in", {e: round(f, 2) for e, f in finger.found.items()})
    bad = [r["episode"] for r in rows if r["status"] != "ok"]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} episodes written; QA in {QA_CSV.relative_to(REPO_ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
