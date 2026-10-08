"""Train/test split (C2.6): seeded, by episode, stratified by category, showcase held out entirely.

    python -m extract.split            # writes data/splits.json; refuses to overwrite a different split

The split is frozen before any method is evaluated. It comes from the manifest alone (not from extraction
results), so it cannot depend on how a method does on an episode.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from capture.common import REPO_ROOT

MANIFEST = REPO_ROOT / "data" / "manifest.csv"
SPLITS = REPO_ROOT / "data" / "splits.json"
SEED = 0
TEST_FRACTION = 0.2
SHOWCASE = "showcase"
DISCARD = "discard"


def read_manifest(path=MANIFEST):
    """Rows of the manifest that are not discards, as (id, category). Short rows (no notes column) are fine."""
    with open(path, newline="") as f:
        rows = [(r["id"].strip(), (r["category"] or "").strip().lower()) for r in csv.DictReader(f)]
    return [(i, c) for i, c in rows if c and c != DISCARD]


def make_split(episodes, seed=SEED, test_fraction=TEST_FRACTION):
    """episodes: iterable of (id, category). Each non-showcase category gives max(1, round(fraction * n)) test episodes."""
    by_cat = {}
    for ep, cat in episodes:
        by_cat.setdefault(cat, []).append(ep)
    rng = np.random.default_rng(seed)
    train, test, per_category = [], [], {}
    for cat in sorted(by_cat):
        if cat == SHOWCASE:
            continue
        ids = sorted(by_cat[cat])                  # sorted first: the seed alone decides, not manifest order
        n_test = max(1, int(round(test_fraction * len(ids))))
        picked = set(rng.permutation(len(ids))[:n_test].tolist())
        te = [e for k, e in enumerate(ids) if k in picked]
        tr = [e for k, e in enumerate(ids) if k not in picked]
        test += te
        train += tr
        per_category[cat] = {"train": len(tr), "test": len(te)}
    return {"seed": seed, "test_fraction": test_fraction, "train": sorted(train), "test": sorted(test),
            "showcase": sorted(by_cat.get(SHOWCASE, [])), "per_category": per_category}


def main(argv=None):
    force = "--force" in (argv or [])
    split = make_split(read_manifest())
    if SPLITS.exists() and not force:
        if json.loads(SPLITS.read_text()) != split:
            raise SystemExit(f"{SPLITS} exists and differs from what the manifest and seed {SEED} give. "
                             f"The split is frozen: do not overwrite it (--force only if no method has run yet).")
        print(f"{SPLITS.name} already matches")
        return 0
    with open(SPLITS, "w", newline="\n") as f:                      # LF on Windows too
        f.write(json.dumps(split, indent=2) + "\n")
    print(f"wrote {SPLITS}: {len(split['train'])} train, {len(split['test'])} test, {len(split['showcase'])} showcase")
    print(split["per_category"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
