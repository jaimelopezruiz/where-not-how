"""Fingertip coverage inside the contact window (T7): how much of each push has a fingertip, and where it is missing.

    python -m extract.finger_coverage --det-cache DIR --tip-cache DIR [--configs crop560:0.3:vid full:0.3:vid]
                                      [--choose-on train] [--out results/finger_coverage.csv]

Works in image space (no fingertip radius needed): the detection of the index fingertip per camera frame
(extract.fingertip.detect_tips) against the contact window of the processed episode. The contact window runs from
the first to the last contact sample of ``data/processed/ep_XXX.npz``. Gaps are measured between detections:
a gap of at most 0.5 s is bridged linearly (extract.fingertip.bridge_gaps); a longer gap stays and the episode
is not usable for hand replay. Fingertip positions are never inferred from the box. Episode-edge gaps
(no detection before the window start, or after its end) are measured against the window edge and tolerated up
to 0.5 s, because the replay moves from standby to the first sample and holds the last one.

``--configs`` are ``full|crop<px> : min_confidence : vid|img`` (for example ``crop560:0.3:vid``). With several, the one with the highest mean
raw coverage on the ``--choose-on`` split (train by default; the test split is never used to pick) is reported
and written; all of them are printed for that split.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from capture.common import REPO_ROOT
from extract import fingertip as F
from extract.detect import RawEpisode
from extract.split import read_manifest

PROCESSED = REPO_ROOT / "data" / "processed"
SPLITS = REPO_ROOT / "data" / "splits.json"
OUT = REPO_ROOT / "results" / "finger_coverage.csv"
COLUMNS = ["episode", "split", "category", "window_s", "frames", "raw_cov", "raw_longest_gap_s", "filled_cov",
           "remaining_longest_gap_s", "usable"]


def parse_config(text):
    kind, conf, mode = text.split(":")
    if not (kind == "full" or kind.startswith("crop") and kind[4:].isdigit()) or mode not in ("vid", "img"):
        raise ValueError(f"config {text!r}: expected full|crop<px>:<min_conf>:vid|img")
    return dict(crop=0 if kind == "full" else int(kind[4:]), min_conf=float(conf), image_mode=mode == "img")


def window_gaps(t, valid, t0, t1, max_gap_s=F.GAP_FILL_S):
    """Coverage and gaps of detections ``valid`` (per camera frame time ``t``) inside the window [t0, t1].

    Returns a dict: frames, raw_cov, raw_longest_gap_s, filled_cov, remaining_longest_gap_s, usable.
    An inner gap is a run of frames without a detection, measured between the detections around it; an edge gap
    runs from the window edge to the first (or from the last) detection when the detections start after (end
    before) the window. Raw lengths count only the part inside the window. After bridging, an inner gap that
    stays counts with its full length (it is longer than ``max_gap_s`` by construction) and an edge gap with
    its part inside the window.
    """
    inside = (t >= t0) & (t <= t1)
    idx = np.flatnonzero(valid)
    gaps = [(t[a - 1], t[b], True) for a, b in F.gap_runs(valid)]                 # (start, end, inner)
    if not len(idx):
        gaps.append((t0, t1, False))
    else:
        if t[idx[0]] > t0:
            gaps.append((t0, t[idx[0]], False))
        if t[idx[-1]] < t1:
            gaps.append((t[idx[-1]], t1, False))
    raw = remaining = 0.0
    for start, end, inner in gaps:
        part = max(0.0, min(end, t1) - max(start, t0))
        raw = max(raw, part)
        if part > 0 and not (inner and end - start <= max_gap_s + 1e-9):
            remaining = max(remaining, end - start if inner else part)
    bridged = F.bridge_gaps(t, np.column_stack([np.where(valid, 0.0, np.nan)] * 2), max_gap_s)[1]
    n = int(inside.sum())
    return dict(frames=n, raw_cov=float(valid[inside].mean()) if n else 0.0, raw_longest_gap_s=raw,
                filled_cov=float(bridged[inside].mean()) if n else 0.0, remaining_longest_gap_s=remaining,
                usable=bool(n) and remaining <= max_gap_s + 1e-9)


def contact_window(ep):
    """(t0, t1) of the first to last contact sample, on the camera clock (which starts at 0), or None."""
    d = np.load(PROCESSED / f"{ep}.npz")
    hit = np.flatnonzero(d["contact"])
    return None if not len(hit) else (float(d["t"][hit[0]]), float(d["t"][hit[-1]]))


def _job(args):
    ep, det_cache, tip_cache, cfg = args
    raw = RawEpisode.load(Path(det_cache) / f"{ep}.npz")
    tips = F.episode_tips(ep, raw, REPO_ROOT / "data" / "raw", cache_dir=tip_cache, **cfg)
    win = contact_window(ep)
    return ep, window_gaps(raw.t, np.isfinite(tips[:, 0]), *win)


def run_config(eps, det_cache, tip_cache, cfg, workers):
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_job, [(e, det_cache, tip_cache, cfg) for e in eps]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det-cache", required=True, help="marker detections (extract.run --cache-dir)")
    ap.add_argument("--tip-cache", required=True, help="where per-episode pixel tips are stored/reused")
    ap.add_argument("--configs", nargs="+", default=["full:0.3:vid"])
    ap.add_argument("--choose-on", default="train", choices=("train", "test", "showcase"))
    ap.add_argument("--choose-n", type=int, default=0, help="compare the configs on this many evenly spaced "
                                                            "episodes of the --choose-on split (0: all of them)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    splits = json.loads(SPLITS.read_text())
    which = {e: s for s in ("train", "test", "showcase") for e in splits[s]}
    cats = dict(read_manifest())
    eps = sorted(cats)
    pick = [e for e in eps if which.get(e) == args.choose_on]
    if args.choose_n and args.choose_n < len(pick):                     # evenly spaced through the sorted ids
        pick = [pick[i] for i in np.linspace(0, len(pick) - 1, args.choose_n).round().astype(int)]

    best, best_score = None, -1.0
    if len(args.configs) > 1:
        for text in args.configs:
            res = run_config(pick, args.det_cache, args.tip_cache, parse_config(text), args.workers)
            score = float(np.mean([r["raw_cov"] for r in res.values()]))
            n_ok = sum(r["usable"] for r in res.values())
            print(f"{text:16s} {args.choose_on}: mean raw coverage in window {score:.3f}, usable {n_ok}/{len(res)}")
            if score > best_score:
                best, best_score = text, score
        print("chosen:", best)
    else:
        best = args.configs[0]
    results = run_config(eps, args.det_cache, args.tip_cache, parse_config(best), args.workers)

    rows = []
    for e in eps:
        t0, t1 = contact_window(e)
        r = {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v for k, v in results[e].items()}
        rows.append(dict(episode=e, split=which.get(e, ""), category=cats[e], window_s=round(t1 - t0, 2), **r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    for s in ("train", "test", "showcase"):
        r = [x for x in rows if x["split"] == s]
        print(f"{s:9s} usable {sum(x['usable'] for x in r)}/{len(r)}   mean raw coverage "
              f"{np.mean([x['raw_cov'] for x in r]):.3f}   mean coverage after bridging "
              f"{np.mean([x['filled_cov'] for x in r]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
