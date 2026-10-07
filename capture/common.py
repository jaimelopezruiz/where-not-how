"""Camera, intrinsics and props helpers shared by the capture scripts."""
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INTRINSICS = REPO_ROOT / "data" / "intrinsics.npz"
DEFAULT_PROPS = REPO_ROOT / "data" / "props.yaml"

# CAP_PROP_AUTO_EXPOSURE is driver-specific. DirectShow (Windows) uses 0.25 for
# manual and 0.75 for auto; V4L2 and most others use 1 for manual and 3 for auto.
_MANUAL_EXPOSURE = {cv2.CAP_DSHOW: 0.25}
_MANUAL_EXPOSURE_DEFAULT = 1.0


def parse_source(text):
    """A bare integer is a camera index; anything else is a video file path."""
    return int(text) if text.isdigit() else text


def open_capture(source, width=None, height=None, fourcc=None, backend=None):
    """Open a camera index or a video file. Returns (cap, backend_used).

    Requested width/height are only requests: webcam drivers silently fall back to
    another mode, so callers must check the size of the frames they actually get.
    """
    if isinstance(source, int):
        backend = cv2.CAP_DSHOW if backend is None else backend
        cap = cv2.VideoCapture(source, backend)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        backend = None
        cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source {source!r}")
    return cap, backend


def lock_exposure(cap, backend, exposure=None, warmup=30):
    """Switch the camera to manual exposure and pin the value. Prints what the driver reports.

    With ``exposure=None`` the value the auto-exposure settled on after ``warmup`` frames is
    written back as the manual value. Setting a property can silently do nothing, so the
    read-back is printed and a mismatch is warned about; it is not proof of a lock. Check
    that the image brightness stays put when the room lighting changes.
    """
    for _ in range(warmup):
        cap.read()
    settled = cap.get(cv2.CAP_PROP_EXPOSURE)
    manual = _MANUAL_EXPOSURE.get(backend, _MANUAL_EXPOSURE_DEFAULT)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
    target = settled if exposure is None else exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, target)
    for _ in range(5):  # let the driver apply it
        cap.read()
    mode = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    value = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"exposure: auto-exposure mode={mode} (manual={manual}), exposure={value} "
          f"(requested {target}, auto had settled on {settled})")
    if mode != manual or abs(value - target) > 1e-6:
        print("WARNING: the driver did not accept the manual exposure setting; "
              "exposure may still be auto. Pass --exposure, or set it in the camera's own settings app.")
    return value


def read_frame(cap):
    ok, frame = cap.read()
    return frame if ok else None


def load_intrinsics(path=DEFAULT_INTRINSICS):
    """Returns (camera_matrix, dist_coeffs, (width, height))."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found: run `python -m capture.calibrate` first (C1.1)")
    with np.load(path) as d:
        w, h = (int(v) for v in d["image_size"])
        return d["camera_matrix"], d["dist_coeffs"], (w, h)


def require_frame_size(frame, expected, what="intrinsics"):
    h, w = frame.shape[:2]
    if (w, h) != tuple(expected):
        raise SystemExit(f"frames are {w}x{h} but the {what} are for {expected[0]}x{expected[1]}; "
                         f"pass --width/--height to match, or recalibrate at this resolution")


def _num(section, key, props_path, required):
    value = section.get(key) if isinstance(section, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    if required:
        raise SystemExit(f"{props_path}: '{key}' is missing or not a positive number; "
                         f"fill it with the measured value (C1.2)")
    return None


# name -> (section, key in data/props.yaml, divisor to SI)
_PROPS = {
    "board_marker_side": ("board", "marker_side_mm", 1000),
    "board_gap": ("board", "gap_mm", 1000),
    "cube_marker_side": ("cube", "marker_side_mm", 1000),
    "cube_side": ("cube", "side_mm", 1000),
    "cube_mass": ("cube", "mass_g", 1000),
}


def load_props(path=DEFAULT_PROPS, require=tuple(_PROPS)):
    """Measured prop dimensions from data/props.yaml, in metres and kilograms.

    The file stores millimetres and grams (what a ruler and scale give). Nothing has a default:
    a value named in ``require`` that is missing or null is an error, never a nominal fallback.
    Other values come back as None when absent.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found (measured prop dimensions, C1.2)")
    raw = yaml.safe_load(path.read_text()) or {}
    out = {}
    for name, (section, key, div) in _PROPS.items():
        value = _num(raw.get(section), key, path, name in require)
        out[name] = None if value is None else value / div
    return out
