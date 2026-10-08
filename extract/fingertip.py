"""Fingertip (C2.3): MediaPipe hand landmark 8 back-projected onto a plane in the table frame.

Only the hand-replay baseline (T7) uses this; the cube trajectory does not depend on it.

The landmark's pixel is turned into a camera ray and intersected with the plane z = z_tip in the table
frame, where z = 0 is the cube-marker plane. The table surface is ``cube.height_mm`` below that (the
marker sits on top of the box), and the fingertip centre is ``finger.tip_radius_mm`` above the table:
    z_tip = tip_radius - cube_height
The fingertip radius is not in the plan's measured props yet. ``data/props.yaml`` needs

    finger:
      tip_radius_mm: <measured>      # half the width of the fingertip where it touches the box

and nothing here has a default for it.

Model: ``data/models/hand_landmarker.task`` (gitignored, 7.8 MB, Google's float16 hand landmarker).
``python -m extract.fingertip --fetch-model`` downloads it from the official MediaPipe model bucket.
"""
import argparse
import urllib.request
import warnings
from pathlib import Path

import cv2
import numpy as np
import yaml

from capture.common import DEFAULT_PROPS, REPO_ROOT
from extract import clean as C

MODEL_PATH = REPO_ROOT / "data" / "models" / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
INDEX_TIP = 8
GAP_FILL_S = 0.5            # detection gaps up to this long (between two detections) are bridged linearly
CROP_TARGET_PX = 800        # a crop smaller than this is scaled up to it


def load_finger_props(path=DEFAULT_PROPS):
    """(tip_radius_m, cube_height_m) from props.yaml. Missing or non-positive values are an error, never a default."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    out = []
    for section, key in (("finger", "tip_radius_mm"), ("cube", "height_mm")):
        v = (raw.get(section) or {}).get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            raise SystemExit(f"{path}: '{section}.{key}' is missing or not a positive number; measure it and add it "
                             f"(the fingertip height sets where the camera ray is cut)")
        out.append(v / 1000)
    return tuple(out)


def fetch_model(dest=MODEL_PATH):
    import hashlib
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(MODEL_URL, timeout=60) as r:
        blob = r.read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != MODEL_SHA256:
        raise SystemExit(f"downloaded model has sha256 {digest}, expected {MODEL_SHA256}; not written")
    dest.write_bytes(blob)
    return dest


def tip_on_plane(pixels, T_cam_board, plane, z_tip, K, dist):
    """Table-frame (x, y, z) of image points cut with the plane z = z_tip. NaN pixels give NaN.

    pixels (N, 2); T_cam_board (N, 4, 4) per-frame board pose; plane from extract.table (marker plane, z = 0).
    """
    out = np.full((len(pixels), 3), np.nan)
    ok = np.isfinite(pixels[:, 0])
    if not ok.any():
        return out
    rays = cv2.undistortPoints(pixels[ok].reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    rays = np.column_stack([rays, np.ones(len(rays))])
    R, t = T_cam_board[ok, :3, :3], T_cam_board[ok, :3, 3]
    n_up = plane.normal_up
    n_cam = R @ n_up                                          # (K, 3)
    d_cam = plane.height + z_tip + np.einsum("ij,ij->i", n_cam, t)          # n.p = d in the board frame, in the camera frame
    s = d_cam / np.einsum("ij,ij->i", n_cam, rays)
    P_cam = rays * s[:, None]
    P_board = np.einsum("kji,kj->ki", R, P_cam - t)           # R^T (p - t)
    A = plane.axes()
    table = P_board @ A.T
    table[:, 2] -= plane.height
    out[ok] = table
    return out


def crop_centres(cube_corners):
    """Image crop centre per frame: the cube marker's centre, the nearest detected one where it was missed."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                    # frames with no marker: all-NaN slices
        c = np.nanmean(cube_corners, axis=1)                               # (N, 2)
    ok = np.flatnonzero(np.isfinite(c[:, 0]))
    if not len(ok):
        raise ValueError("cube marker never detected; there is nothing to centre the crop on")
    near = ok[np.abs(np.arange(len(c))[:, None] - ok[None]).argmin(axis=1)]
    return c[near]


def crop_window(centre, size, frame_w, frame_h):
    """Top-left corner of a ``size`` px square around ``centre``, slid back inside the frame."""
    x0 = int(round(min(max(centre[0] - size / 2, 0), frame_w - size)))
    y0 = int(round(min(max(centre[1] - size / 2, 0), frame_h - size)))
    return x0, y0


def detect_tips(video_path, timestamps, model_path=MODEL_PATH, min_conf=0.5, centres=None, crop_px=400,
                upscale=2.0, image_mode=False):
    """Pixel position of the index fingertip in every frame, (N, 2), NaN where no hand was found.

    With ``centres`` (N, 2), MediaPipe sees a ``crop_px`` square around that point, scaled up by ``upscale``,
    instead of the whole frame: the fingertip that pushes the box is a small part of a 1280x720 image
    (landmarks come back in full-frame pixels either way). ``image_mode`` detects each frame on its own
    instead of tracking the hand from the previous one.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    if not Path(model_path).exists():
        raise SystemExit(f"{model_path} not found: run `python -m extract.fingertip --fetch-model`")
    mode = vision.RunningMode.IMAGE if image_mode else vision.RunningMode.VIDEO
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), running_mode=mode, num_hands=1,
        min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf, min_tracking_confidence=min_conf)
    cap = cv2.VideoCapture(str(video_path))
    n = len(timestamps)
    out = np.full((n, 2), np.nan)
    last_ms = -1
    with vision.HandLandmarker.create_from_options(opts) as lm:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            x0 = y0 = 0
            if centres is not None:
                x0, y0 = crop_window(centres[i], crop_px, w, h)
                frame = cv2.resize(frame[y0:y0 + crop_px, x0:x0 + crop_px], None, fx=upscale, fy=upscale,
                                   interpolation=cv2.INTER_CUBIC)
                h, w = crop_px, crop_px
            ms = max(int(round(timestamps[i] * 1000)), last_ms + 1)         # VIDEO mode needs strictly increasing ms
            last_ms = ms
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lm.detect(img) if image_mode else lm.detect_for_video(img, ms)
            if res.hand_landmarks:
                p = res.hand_landmarks[0][INDEX_TIP]
                out[i] = (x0 + p.x * w, y0 + p.y * h)
    cap.release()
    return out


def gap_runs(valid):
    """[start, stop) index pairs of the runs of invalid frames that have a valid frame on both sides."""
    return [(a, b) for a, b in C.segments(~valid) if a > 0 and b < len(valid)]


def bridge_gaps(t, xy, max_gap_s=GAP_FILL_S):
    """Linearly fill the gaps between two detections that are at most ``max_gap_s`` apart (time between the
    detections on either side). Nothing is ever extrapolated at the ends or filled from anything but the
    fingertip itself. Returns (xy with NaN left in longer gaps, mask of frames now valid)."""
    valid = np.isfinite(xy[:, 0])
    out = np.where(valid[:, None], xy, np.nan)
    ok = valid.copy()
    for a, b in gap_runs(valid):
        if t[b] - t[a - 1] <= max_gap_s + 1e-9:
            for c in range(2):
                out[a:b, c] = np.interp(t[a:b], [t[a - 1], t[b]], [xy[a - 1, c], xy[b, c]])
            ok[a:b] = True
    return out, ok


def onto_grid(t, xy, grid, max_gap_s=GAP_FILL_S):
    """Bridge gaps of up to ``max_gap_s``, smooth like the cube, interpolate onto ``grid``; NaN elsewhere."""
    ok = np.isfinite(xy[:, 0])
    out = np.full((len(grid), 2), np.nan)
    if ok.sum() < 2:
        return out
    filled, ok = bridge_gaps(t, xy, max_gap_s)
    filled = np.nan_to_num(filled)
    smoothed = C.smooth(t, filled, ok)
    for a, b in C.segments(ok):
        inside = (grid >= t[a] - 1e-9) & (grid <= t[b - 1] + 1e-9)
        for c in range(2):
            out[inside, c] = np.interp(grid[inside], t[a:b], smoothed[a:b, c])
    return out


def episode_tips(ep, raw, video_dir, model_path=MODEL_PATH, min_conf=0.5, crop=0, image_mode=False,
                 cache_dir=None):
    """Pixel tips of one episode, from ``cache_dir`` when a run with the same settings stored them there.

    ``crop`` is the side in px of the square window around the box that MediaPipe sees (0: the whole frame).
    """
    tag = f"{ep}_tips_c{min_conf:g}_{f'crop{int(crop)}' if crop else 'full'}_{'img' if image_mode else 'vid'}.npy"
    cache = Path(cache_dir) / tag if cache_dir else None
    if cache is not None and cache.exists():
        return np.load(cache)
    tips = detect_tips(Path(video_dir) / f"{ep}.avi", raw.t, model_path, min_conf,
                       centres=crop_centres(raw.cube_corners) if crop else None, crop_px=int(crop),
                       upscale=max(1.0, CROP_TARGET_PX / crop) if crop else 1.0, image_mode=image_mode)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, tips)
    return tips


class FingerTracker:
    """Callable for ``extract.run.run(finger=...)``: finger_xy on the episode's output grid."""

    def __init__(self, K, dist, model_path=MODEL_PATH, props_path=DEFAULT_PROPS, video_dir=None, min_conf=0.5,
                 crop=0, image_mode=False, cache_dir=None):
        from extract.run import RAW_DIR
        self.K, self.dist, self.model_path = K, dist, model_path
        radius, height = load_finger_props(props_path)
        self.z_tip = radius - height
        self.video_dir = Path(video_dir or RAW_DIR)
        self.detect_kw = dict(min_conf=min_conf, crop=crop, image_mode=image_mode, cache_dir=cache_dir)
        self.found = {}

    def __call__(self, ep, raw, track, plane, t_grid_abs):
        tips = episode_tips(ep, raw, self.video_dir, self.model_path, **self.detect_kw)
        self.found[ep] = float(np.isfinite(tips[:, 0]).mean())
        xyz = tip_on_plane(tips, track.T_cam_board, plane, self.z_tip, self.K, self.dist)
        return onto_grid(raw.t, xyz[:, :2], t_grid_abs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch-model", action="store_true", help=f"download {MODEL_PATH.name} into data/models/")
    args = ap.parse_args(argv)
    if args.fetch_model:
        print("wrote", fetch_model())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
