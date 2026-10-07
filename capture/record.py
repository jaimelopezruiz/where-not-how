"""Record demonstration episodes (C1.3): video, per-frame timestamps, manifest row, metadata.

    python -m capture.record

Defaults are the capture settings fixed in C1.2 (--source 2 --width 1280 --height 720 --exposure -6
--fourcc MJPG). Keys in the preview window (it must have focus):

    s c t m l   category: straight, curve, turn, multi, showcase. Sticky until changed.
    d           category "discard": still logged in the manifest, nothing is ever deleted.
    n           type notes for the following episodes in the terminal.
    SPACE       start / stop an episode (needs a category). q or ESC quits (stopping any episode first).

Per episode, in --out-dir (data/raw, gitignored):
    ep_XXX.avi        MJPG, written by OpenCV's built-in encoder at --quality 100
    ep_XXX_t.csv      frame_index, perf_counter_s: time each frame was read from the camera
    ep_XXX_meta.json  source, size, camera format, exposure read-back, measured fps, sha256 of intrinsics.npz, props
and one row (id, category, notes) appended to data/manifest.csv.

Record with the hand out of frame for 1 s at the start and end of each clip. Refuses to run when the frame
size differs from data/intrinsics.npz. --test writes test_XXX.* instead and skips the manifest.
"""
import argparse
import csv
import hashlib
import json
import re
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from capture.common import (DEFAULT_INTRINSICS, DEFAULT_PROPS, REPO_ROOT, load_intrinsics, lock_exposure,
                            open_capture, parse_source, read_frame, require_frame_size)

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest.csv"
MANIFEST_FIELDS = ["id", "category", "notes"]
CATEGORIES = {"s": "straight", "c": "curve", "t": "turn", "m": "multi", "l": "showcase", "d": "discard"}

# cv2.VideoWriter(path, fourcc, ...) picks the FFMPEG backend, whose MJPEG quality cannot be set from OpenCV:
# reading its clips back shifted the detected pose by up to 6 mm and 1.7 deg. The built-in encoder takes a
# quality (0-100) through writer.set(); at 100 the shift was 0.1 mm and 0.01 deg (decision log, C1.3).
WRITER_BACKEND = cv2.CAP_OPENCV_MJPEG
WRITER_BACKEND_NAME = "CV_MJPEG"
MIN_FPS_SAMPLES = 10
ESTIMATE_FRAMES = 5


def open_writer(path, size, fps, quality):
    """Open the MJPG .avi writer and confirm the backend and quality are what was asked for."""
    w = cv2.VideoWriter(str(path), WRITER_BACKEND, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), tuple(size))
    if not w.isOpened():
        raise SystemExit(f"cannot open a video writer for {path}")
    ok = w.set(cv2.VIDEOWRITER_PROP_QUALITY, float(quality))
    if w.getBackendName() != WRITER_BACKEND_NAME or not ok or w.get(cv2.VIDEOWRITER_PROP_QUALITY) != quality:
        w.release()
        raise SystemExit(f"video writer is {w.getBackendName()} and did not accept quality {quality}; "
                         f"clips would be recorded at an uncontrolled quality")
    return w


def estimate_mb_per_s(frames, size, fps, quality):
    """Disk rate of the real writer, measured by encoding a few preview frames to a temporary file."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "estimate.avi"
        w = open_writer(path, size, fps, quality)
        for f in frames:
            w.write(f)
        w.release()
        return path.stat().st_size / len(frames) * fps / 1e6


def next_episode_stem(out_dir, manifest, prefix):
    """Next free ``prefix_NNN``: one past the highest number among files in out_dir and manifest rows."""
    pattern = re.compile(rf"^{prefix}_(\d+)")
    seen = [int(m.group(1)) for p in Path(out_dir).glob(f"{prefix}_*") if (m := pattern.match(p.name))]
    manifest = Path(manifest)
    if prefix == "ep" and manifest.exists():
        with open(manifest, newline="") as f:
            seen += [int(m.group(1)) for row in csv.DictReader(f) if (m := pattern.match(row.get("id") or ""))]
    return f"{prefix}_{max(seen, default=-1) + 1:03d}"


def append_manifest(path, ep_id, category, notes):
    """Append one row, writing the header when the file is new or empty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        if not new and not path.read_bytes().endswith(b"\n"):
            f.write("\n")                                    # a hand-edited file may lack the final newline
        w = csv.writer(f)
        if new:
            w.writerow(MANIFEST_FIELDS)
        w.writerow([ep_id, category, notes])


def fourcc_text(value):
    return "".join(chr((int(value) >> (8 * i)) & 0xFF) for i in range(4))


def timing_summary(ts):
    """Frames, duration, mean fps, largest gap and how many gaps exceed 1.5x the median (a dropped frame)."""
    ts = np.asarray(ts, float)
    dt = np.diff(ts)
    if len(dt) == 0:
        return {"frames": len(ts), "duration_s": 0.0, "fps": 0.0, "max_gap_ms": 0.0, "long_gaps": 0}
    return {"frames": len(ts), "duration_s": float(ts[-1] - ts[0]), "fps": float((len(ts) - 1) / (ts[-1] - ts[0])),
            "max_gap_ms": float(dt.max() * 1000), "long_gaps": int(np.sum(dt > 1.5 * np.median(dt)))}


class Episode:
    """One recording: the video, its timestamp CSV and a meta file. Files are created exclusively, never overwritten."""

    def __init__(self, out_dir, stem, size, fps, quality):
        self.out_dir, self.stem = Path(out_dir), stem
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.avi = self.out_dir / f"{stem}.avi"
        self.csv_path = self.out_dir / f"{stem}_t.csv"
        self.meta_path = self.out_dir / f"{stem}_meta.json"
        for p in (self.avi, self.csv_path, self.meta_path):
            if p.exists():
                raise SystemExit(f"{p} already exists; refusing to overwrite")
        self.writer = open_writer(self.avi, size, fps, quality)
        self._csv_file = open(self.csv_path, "x", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["frame_index", "perf_counter_s"])
        self.ts, self.dropped = [], 0

    def add(self, frame, t):
        self.writer.write(frame)
        self._csv.writerow([len(self.ts), f"{t:.9f}"])
        self.ts.append(t)

    def close(self):
        self.writer.release()
        self._csv_file.close()
        return timing_summary(self.ts)


def video_frame_count(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw_hud(frame, category, test, fps, ep, stem, min_fps, fps_ok):
    vis = frame.copy()                                       # never draw on the frame that is recorded
    h = vis.shape[0]
    label = "TEST" if test else (category or "press s/c/t/m/l for a category")
    cv2.putText(vis, f"{label}   {stem}   {fps:4.1f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 200, 0) if fps_ok else (0, 0, 255), 2)
    if ep is not None:
        elapsed = ep.ts[-1] - ep.ts[0] if ep.ts else 0.0
        cv2.circle(vis, (22, 62), 9, (0, 0, 255), -1)
        cv2.putText(vis, f"REC {elapsed:5.1f}s  {len(ep.ts)} frames", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        if elapsed < 1.0:
            cv2.putText(vis, "HANDS OUT OF FRAME", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif not fps_ok:
        cv2.putText(vis, f"fps below {min_fps:g}: REC disabled (camera format?)", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(vis, "SPACE rec/stop | s c t m l d category | n notes | q quit", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return vis


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="2", help="camera index (default 2: the C270) or a video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fourcc", default="MJPG", help="camera pixel format (default MJPG; YUY2 gives 7.5 fps)")
    ap.add_argument("--exposure", type=float, default=-6.0, help="manual exposure value (default -6)")
    ap.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS))
    ap.add_argument("--props", default=str(DEFAULT_PROPS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--quality", type=int, default=100, help="MJPG quality 0-100 (default 100)")
    ap.add_argument("--min-fps", type=float, default=20.0, help="refuse to record below this measured fps")
    ap.add_argument("--max-seconds", type=float, help="stop each episode automatically after this long")
    ap.add_argument("--category", choices=sorted(set(CATEGORIES.values())), help="start with this category chosen")
    ap.add_argument("--notes", default="", help="notes written to the manifest for each episode")
    ap.add_argument("--test", action="store_true", help="write test_XXX.* files, no category, no manifest row")
    ap.add_argument("--auto-start", action="store_true", help="start an episode on the first frame (no keypress)")
    ap.add_argument("--no-display", action="store_true", help="no window, so no keys: use with --auto-start")
    ap.add_argument("--max-frames", type=int, help="stop after reading this many frames")
    args = ap.parse_args(argv)

    K, dist, intr_size = load_intrinsics(args.intrinsics)
    source = parse_source(args.source)
    is_camera = isinstance(source, int)
    cap, backend = open_capture(source, args.width, args.height, args.fourcc)
    exposure = None
    if is_camera:
        exposure = {"requested": args.exposure, "readback": lock_exposure(cap, backend, args.exposure),
                    "auto_exposure_mode": cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}
    camera_fourcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    prefix = "test" if args.test else "ep"
    category = args.category
    notes = args.notes
    ep, ep_info, next_stem = None, None, None
    summaries = []
    recent, probe_frames = deque(maxlen=30), []
    size = None
    total = failures = 0

    def fps_now():
        return (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 and recent[-1] > recent[0] else 0.0

    def start_episode():
        nonlocal ep, ep_info
        if not args.test and category is None:
            print("not recording: choose a category first (s c t m l, or d to discard)")
            return
        fps = fps_now()
        if is_camera and len(recent) < MIN_FPS_SAMPLES:
            print(f"not recording yet: still measuring fps ({len(recent)}/{MIN_FPS_SAMPLES} frames)")
            return
        if is_camera and fps < args.min_fps:
            print(f"not recording: measured {fps:.1f} fps is below --min-fps {args.min_fps:g}; "
                  f"the camera is probably not delivering MJPG (check --fourcc)")
            return
        header_fps = round(fps, 1) if is_camera else file_fps
        stem = next_episode_stem(args.out_dir, args.manifest, prefix)
        ep = Episode(args.out_dir, stem, size, header_fps, args.quality)
        ep_info = {"category": "test" if args.test else category, "notes": notes, "start_utc": datetime.now(timezone.utc).isoformat()}
        print(f"recording {stem} ({ep_info['category']})")

    def stop_episode():
        nonlocal ep, ep_info, next_stem
        summary = ep.close()
        n_video = video_frame_count(ep.avi)
        meta = {
            "id": ep.stem, "category": ep_info["category"], "notes": ep_info["notes"], "start_utc": ep_info["start_utc"],
            "source": args.source, "frame_size": list(size),
            "camera_fourcc_requested": args.fourcc if is_camera else None, "camera_fourcc_readback": camera_fourcc,
            "exposure": exposure, "measured_fps": summary["fps"], "frames": summary["frames"],
            "writer": {"backend": WRITER_BACKEND_NAME, "codec": "MJPG", "quality": args.quality},
            "intrinsics_file": Path(args.intrinsics).name, "intrinsics_sha256": sha256_of(args.intrinsics),
            "props": yaml.safe_load(Path(args.props).read_text()) if Path(args.props).exists() else None,
            "opencv": cv2.__version__,
        }
        with open(ep.meta_path, "x") as f:
            json.dump(meta, f, indent=2)
        if not args.test:
            append_manifest(args.manifest, ep.stem, ep_info["category"], ep_info["notes"])
        mb = ep.avi.stat().st_size / 1e6
        print(f"saved {ep.stem}: {summary['frames']} frames in {summary['duration_s']:.1f} s = {summary['fps']:.1f} fps, "
              f"max gap {summary['max_gap_ms']:.0f} ms, gaps over 1.5x median {summary['long_gaps']}, "
              f"{mb:.1f} MB; video holds {n_video} frames vs {summary['frames']} timestamps"
              + (f"; {ep.dropped} failed reads" if ep.dropped else ""))
        if n_video != summary["frames"]:
            print("WARNING: video frame count differs from the timestamp rows")
        summaries.append({**summary, "id": ep.stem, "video_frames": n_video})
        ep = ep_info = next_stem = None

    try:
        while True:
            frame = read_frame(cap)
            t = time.perf_counter()
            if frame is None:
                failures += 1
                if ep is not None and is_camera:         # end of a file source is not a failed read
                    ep.dropped += 1
                if not is_camera:
                    print("end of video file")
                    break
                if failures >= 30:
                    print(f"stopping: {failures} camera reads in a row returned no frame (camera busy or unplugged?)")
                    break
                continue
            failures = 0
            if size is None:
                require_frame_size(frame, intr_size)         # refuse before anything is written
                size = (frame.shape[1], frame.shape[0])
                print(f"camera delivers {size[0]}x{size[1]}, format {camera_fourcc!r}")
            elif frame.shape[:2] != (size[1], size[0]):
                raise SystemExit(f"frame size changed mid-session to {frame.shape[1]}x{frame.shape[0]}")
            total += 1
            recent.append(t)
            if len(probe_frames) < ESTIMATE_FRAMES:
                probe_frames.append(frame)
                if len(probe_frames) == ESTIMATE_FRAMES:
                    rate = estimate_mb_per_s(probe_frames, size, 30.0, args.quality)
                    print(f"expected disk rate at quality {args.quality}, 30 fps: about {rate:.1f} MB/s "
                          f"({rate * 20:.0f} MB per 20 s episode)")
            if ep is None and args.auto_start and not summaries:
                start_episode()
            if ep is not None:
                ep.add(frame, t)
                if args.max_seconds and t - ep.ts[0] >= args.max_seconds:
                    stop_episode()
            key = -1
            if not args.no_display:
                fps = fps_now()
                if ep is None and next_stem is None:
                    next_stem = next_episode_stem(args.out_dir, args.manifest, prefix)
                cv2.imshow("record", draw_hud(frame, category, args.test, fps, ep, ep.stem if ep else next_stem,
                                              args.min_fps, not is_camera or fps >= args.min_fps))
                key = cv2.waitKey(1) & 0xFF
            ch = chr(key) if 0 <= key < 128 else ""
            if ch == " ":
                stop_episode() if ep is not None else start_episode()
            elif ch in CATEGORIES:
                if ep is not None:
                    print("category is fixed while recording")
                else:
                    category = CATEGORIES[ch]
                    print(f"category: {category}")
            elif ch == "n" and ep is None:
                notes = input("notes for the following episodes: ").strip()
            elif key in (27, ord("q")):
                print("quit (q/ESC)")
                break
            if args.max_frames and total >= args.max_frames:
                print(f"stopping: --max-frames {args.max_frames} reached")
                break
    finally:
        if ep is not None:
            stop_episode()                                   # never leave an unfinished clip without its index
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return summaries


if __name__ == "__main__":
    main()
