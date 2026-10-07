"""Webcam intrinsic calibration from a checkerboard (C1.1).

Live capture (SPACE saves a frame, but only while the full board is detected):
    python -m capture.calibrate --square-mm 25 --width 1280 --height 720

Recompute from saved frames without the camera:
    python -m capture.calibrate --square-mm 25 --from-frames data/raw/calib

Keys: SPACE save frame | u undo last | c calibrate and write | q or ESC quit.
The checkerboard is 9x6 inner corners by default (--corners). Measure one square with a ruler
(on a monitor, measure the displayed square) and pass it as --square-mm.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from capture.common import (DEFAULT_INTRINSICS, REPO_ROOT, lock_exposure, open_capture, parse_source,
                            read_frame)

MIN_FRAMES = 10            # below this calibrateCamera is not worth running
RECOMMENDED_FRAMES = 20    # plan: 20-30 sharp frames
ACCEPT_RMS_PX = 1.0        # plan: accept if reprojection error < 1 px
DEFAULT_FRAMES_DIR = REPO_ROOT / "data" / "raw" / "calib"   # gitignored with data/raw


def parse_corners(text):
    cols, rows = (int(v) for v in text.lower().split("x"))
    return cols, rows


def find_corners(gray, pattern):
    """Detect inner corners with the sector-based detector (sub-pixel accurate, so no cornerSubPix)."""
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return (True, corners) if found else (False, None)


def sharpness(gray, corners):
    """Variance of the Laplacian inside the board's bounding box; higher is sharper. Only comparable
    between frames of the same camera and lighting."""
    x, y, w, h = cv2.boundingRect(corners.reshape(-1, 2).astype(np.float32))
    roi = gray[max(y, 0):y + h, max(x, 0):x + w]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


def object_points(pattern, square_m):
    grid = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_m
    return grid


def calibrate(all_corners, pattern, square_m, image_size, fix_k3=False):
    """Run cv2.calibrateCamera. Returns a dict that is saved as intrinsics.npz.

    image_size is (width, height). per_view_rms is the reprojection error of each frame in px,
    so a bad frame can be found and deleted from the frames directory before recomputing.
    fix_k3 holds the third radial term at zero, which stops it absorbing noise when there are few frames.
    """
    obj = object_points(pattern, square_m)
    obj_points = [obj] * len(all_corners)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, all_corners, image_size, None, None, flags=cv2.CALIB_FIX_K3 if fix_k3 else 0)
    per_view = []
    for o, c, r, t in zip(obj_points, all_corners, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        # reshape both: projectPoints gives (N, 1, 2) but the detector's corner shape differs
        # between OpenCV versions (5.0 returns (N, 2)), and subtracting them unreshaped broadcasts
        err = proj.reshape(-1, 2) - np.asarray(c).reshape(-1, 2)
        per_view.append(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))))
    return {
        "camera_matrix": K,
        "dist_coeffs": dist.ravel(),
        "image_size": np.array(image_size),
        "rms": float(rms),
        "per_view_rms": np.array(per_view),
        "square_m": square_m,
        "pattern": np.array(pattern),
    }


def save_intrinsics(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)


def report(result, path=None):
    K, d = result["camera_matrix"], result["dist_coeffs"]
    w, h = result["image_size"]
    pv = result["per_view_rms"]
    print(f"\nframes used: {len(pv)}   image size: {w}x{h}")
    print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}   (image centre {w / 2:.1f}, {h / 2:.1f})")
    print("dist (k1 k2 p1 p2 k3):", np.round(d, 4))
    worst = np.argsort(pv)[::-1][:3]
    print("worst frames (index: rms px):", ", ".join(f"{i}: {pv[i]:.2f}" for i in worst))
    verdict = "ACCEPT" if result["rms"] < ACCEPT_RMS_PX else "REJECT (recapture or drop the worst frames)"
    print(f"reprojection error (RMS): {result['rms']:.3f} px  -> {verdict} (threshold {ACCEPT_RMS_PX} px)")
    if len(pv) < RECOMMENDED_FRAMES:
        print(f"note: {len(pv)} frames is below the recommended {RECOMMENDED_FRAMES}")
    if path:
        print(f"written: {path}")


def run_offline(args, pattern):
    paths = sorted(p for p in Path(args.from_frames).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not paths:
        raise SystemExit(f"no images in {args.from_frames}")
    corners, size = [], None
    for p in paths:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"skip {p.name}: unreadable")
            continue
        this = (gray.shape[1], gray.shape[0])
        if size is not None and this != size:
            raise SystemExit(f"{p.name} is {this[0]}x{this[1]}, earlier frames were {size[0]}x{size[1]}")
        size = this
        found, c = find_corners(gray, pattern)
        if found:
            corners.append(c)
        else:
            print(f"skip {p.name}: board not found")
    if len(corners) < MIN_FRAMES:
        raise SystemExit(f"only {len(corners)} usable frames; need at least {MIN_FRAMES}")
    result = calibrate(corners, pattern, args.square_mm / 1000, size, args.fix_k3)
    save_intrinsics(result, args.out)
    report(result, args.out)
    return result


def draw_hud(frame, found, corners, pattern, saved, sharp, size):
    vis = frame.copy()
    for c in saved:                                  # coverage so far: one dot per saved corner
        for p in c.reshape(-1, 2)[::7]:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (255, 160, 0), -1)
    if found:
        cv2.drawChessboardCorners(vis, pattern, corners, True)
    status = f"board found  sharpness {sharp:.0f}  SPACE to save" if found else "board NOT found"
    colour = (0, 200, 0) if found else (0, 0, 255)
    cv2.putText(vis, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(vis, f"saved {len(saved)}  (aim {RECOMMENDED_FRAMES}-30)  {size[0]}x{size[1]}  u undo  c calibrate  q quit",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def run_live(args, pattern):
    cap, backend = open_capture(parse_source(args.source), args.width, args.height, args.fourcc)
    if isinstance(parse_source(args.source), int):
        lock_exposure(cap, backend, args.exposure)
    frame_dir = Path(args.frames_dir)
    saved, saved_frames, size, result = [], [], None, None
    print(__doc__.split("\n\n")[0])
    while True:
        frame = read_frame(cap)
        if frame is None:
            print("camera returned no frame")
            break
        this = (frame.shape[1], frame.shape[0])
        if size is None:
            size = this
            if (args.width and args.width != size[0]) or (args.height and args.height != size[1]):
                print(f"WARNING: asked for {args.width}x{args.height} but the camera delivers {size[0]}x{size[1]}")
            print(f"capturing at {size[0]}x{size[1]}; the recording resolution must be identical")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern)
        sharp = sharpness(gray, corners) if found else 0.0
        cv2.imshow("calibrate", draw_hud(frame, found, corners, pattern, saved, sharp, size))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            if found:                                # frames without a detected board are never saved
                saved.append(corners)
                saved_frames.append(frame)
                print(f"saved frame {len(saved)} (sharpness {sharp:.0f})")
            else:
                print("not saved: board not found")
        elif key == ord("u") and saved:
            saved.pop(), saved_frames.pop()
            print(f"removed last frame; {len(saved)} left")
        elif key == ord("c"):
            if len(saved) < MIN_FRAMES:
                print(f"need at least {MIN_FRAMES} frames, have {len(saved)}")
                continue
            result = calibrate(saved, pattern, args.square_mm / 1000, size, args.fix_k3)
            save_intrinsics(result, args.out)
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(saved_frames):
                cv2.imwrite(str(frame_dir / f"calib_{i:03d}.png"), f)
            report(result, args.out)
            print(f"frames kept in {frame_dir}; delete bad ones and rerun with --from-frames to recompute")
    cap.release()
    cv2.destroyAllWindows()
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--square-mm", type=float, required=True, help="measured side of one checkerboard square, mm")
    ap.add_argument("--corners", default="9x6", help="inner corners as COLSxROWS (default 9x6)")
    ap.add_argument("--source", default="0", help="camera index (default 0)")
    ap.add_argument("--width", type=int, help="requested frame width; check the printed actual size")
    ap.add_argument("--height", type=int, help="requested frame height")
    ap.add_argument("--fourcc", help="pixel format request, e.g. MJPG (higher frame rate on USB 2)")
    ap.add_argument("--exposure", type=float, help="manual exposure value (driver units; DirectShow is log2 seconds). "
                    "Default: pin whatever auto-exposure settled on")
    ap.add_argument("--fix-k3", action="store_true", help="hold the k3 radial term at zero (try this if k2/k3 come out large)")
    ap.add_argument("--out", default=str(DEFAULT_INTRINSICS), help="intrinsics file to write")
    ap.add_argument("--frames-dir", default=str(DEFAULT_FRAMES_DIR), help="where accepted frames are stored")
    ap.add_argument("--from-frames", help="calibrate from a directory of saved images instead of the camera")
    args = ap.parse_args(argv)
    pattern = parse_corners(args.corners)
    return run_offline(args, pattern) if args.from_frames else run_live(args, pattern)


if __name__ == "__main__":
    main()
