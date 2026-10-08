"""Schema of ``data/processed/ep_XXX.npz`` (C2.5) and the validator every real episode must pass.

    t               (M,)   float64  s, strictly increasing, 0 at the first sample, uniform at the control rate
    cube_xy_yaw     (M, 3) float64  m, m, rad: cube pose in the table frame (z up, yaw counter-clockwise, unwrapped)
    finger_xy       (M, 2) float64  m: fingertip in the table frame; NaN where the hand was not found
    contact         (M,)   bool     the cube is being pushed
    cube_marker_side ()    float64  m: marker edge recovered from the image alone (a size check, not a command)

Downstream code reads the first four. Run on every file in data/processed with ``python -m extract.schema``.
"""
import sys
from pathlib import Path

import numpy as np

from capture.common import REPO_ROOT, load_props
from extract.clean import CONTROL_HZ

PROCESSED = REPO_ROOT / "data" / "processed"
REQUIRED = {"t": 1, "cube_xy_yaw": 2, "finger_xy": 2, "contact": 1, "cube_marker_side": 0}
EXTENT_MAX = 1.5           # m: positions beyond this are millimetres or centimetres, not metres
MIN_PATH = 0.05            # m: a push moves the box at least this far
SIZE_TOL = 0.05            # recovered marker side vs measured, fraction


def problems(data, props=None):
    """List of what is wrong with one episode (a mapping of arrays); empty when it is valid."""
    props = props or load_props(require=("cube_marker_side",))
    out = []
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return [f"missing keys {missing}"]
    t, pose, finger, contact = (np.asarray(data[k]) for k in ("t", "cube_xy_yaw", "finger_xy", "contact"))
    m = len(t)
    if t.ndim != 1 or m < 10:
        return [f"t must be 1-D with at least 10 samples, got shape {t.shape}"]
    for name, arr, cols in (("cube_xy_yaw", pose, 3), ("finger_xy", finger, 2)):
        if arr.shape != (m, cols):
            out.append(f"{name} shape {arr.shape}, expected {(m, cols)}")
    if contact.shape != (m,) or contact.dtype != bool:
        out.append(f"contact must be bool with shape {(m,)}, got {contact.dtype} {contact.shape}")
    for name in ("t", "cube_xy_yaw", "finger_xy"):
        if np.asarray(data[name]).dtype != np.float64:
            out.append(f"{name} dtype {np.asarray(data[name]).dtype}, expected float64")
    if out:
        return out
    if not np.all(np.isfinite(t)) or t[0] != 0 or np.any(np.diff(t) <= 0):
        out.append("t must start at 0 and be strictly increasing")
    elif np.abs(np.diff(t) - 1 / CONTROL_HZ).max() > 1e-6:
        out.append(f"t is not uniform at {CONTROL_HZ} Hz")
    if not np.all(np.isfinite(pose)):
        out.append("cube_xy_yaw has NaN or inf (a gap longer than the interpolation limit)")
    else:
        if np.abs(pose[:, :2]).max() > EXTENT_MAX:
            out.append(f"cube position reaches {np.abs(pose[:, :2]).max():.2f}: not in metres")
        if np.hypot(*np.diff(pose[:, :2], axis=0).T).sum() < MIN_PATH:
            out.append("cube moves less than 5 cm in total")
        if np.abs(np.diff(pose[:, 2])).max() > np.pi:
            out.append("yaw jumps by more than pi between samples: not unwrapped")
    seen = np.isfinite(finger)
    if seen.any() and np.abs(finger[seen]).max() > EXTENT_MAX:
        out.append("finger position not in metres")
    if not contact.any():
        out.append("no contact samples")
    side, want = float(data["cube_marker_side"]), props["cube_marker_side"]
    if not np.isfinite(side) or abs(side / want - 1) > SIZE_TOL:
        out.append(f"recovered marker side {side * 1000:.1f} mm vs measured {want * 1000:.1f} mm (> {SIZE_TOL:.0%})")
    return out


def load_episode(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def validate_file(path, props=None):
    return problems(load_episode(path), props)


def main(argv=None):
    paths = sorted(Path(argv[0] if argv else PROCESSED).glob("ep_*.npz"))
    if not paths:
        raise SystemExit(f"no ep_*.npz in {PROCESSED}")
    bad = 0
    for p in paths:
        errs = validate_file(p)
        bad += bool(errs)
        print(f"{p.name}: {'ok' if not errs else '; '.join(errs)}")
    print(f"{len(paths) - bad}/{len(paths)} valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
