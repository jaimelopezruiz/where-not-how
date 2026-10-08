"""C8.1: showcase pipeline. A controller on the held-out letter traces, side by side with the human video.

    python -m scripts.showcase --controller scripted
    python -m scripts.showcase --run runs/residual_s0 [--model <checkpoint .zip>] [--method residual]

1. Feasibility: each showcase trace goes through the frozen workspace map and the check C3.4 used (every
   point: box clear of the base and the pusher behind it within reach of the FK sample). Written to
   results/showcase_feasibility.csv. The 200 000-sample FK set takes ~3 min, so it is only recomputed when
   that file is missing or with --recheck-feasibility (--skip-feasibility to bypass). Nothing is refitted.
2. The controller runs on every showcase episode through eval.report.evaluate(until_done=True), with a
   recorder around the env that keeps a panel-sized render of every step. Writes results/showcase_<method>.csv,
   _summary.txt, _overlay.png (all four overlaid) and _letters.png (one panel per letter, upright).
3. One GIF per episode, results/showcase_<method>_<episode>.gif: left the human video, right the sim, with the
   reference path dashed on the sim. 4. results/showcase_<method>.gif: the four pairs in a 2x2 grid.

Clock. Both panels share one clock at real speed (the sim steps at 20 Hz = real time). Time 0 is the start of
the push: the first moment the box is more than 5 mm from where it began, found the same way in the extracted
trajectory (human video) and in the simulated box. The GIF starts --lead seconds before that, ends --tail
seconds after the later of the two pushes ends (box within 5 mm of where it finishes), and the panel that
finishes first holds its last frame. The recorded video is read at its CSV timestamps.

Orientation. The raw image has the table frame's x to the right and y up (the board reads upright), and the sim
overhead camera has x up and y to the left. The two are rotations of each other, never mirrored. Letters are
shown as the author drew them: L, U, S seen from the near end of the table, board at the top of the panel
(real image rotated 90 deg counter-clockwise, sim render as is); the C was traced to read from the board side,
board at the bottom (real image rotated 90 deg clockwise, sim render rotated 180 deg). See VIEWS.

Crop. Real: a square around the track of the box marker (ArUco detection every 3rd frame) plus a margin of
about half a box side. Sim: a square around the reference path plus 6 cm. The sim path is the human path
scaled by the workspace map (0.8) and shifted, so the two panels show similar but not identical extents.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from capture.common import REPO_ROOT, load_props
from control.workspace import WORKSPACE_MAP_PATH, load_workspace_map, sample_reachable_xy
from eval.report import category_table, evaluate, plot_overlay, write_results_csv
from extract.detect import read_timestamps
from sim.push_env import PROCESSED_DIR, SPLITS_PATH, PushTrackEnv

# episode -> (letter, view). "ccw": board at the top of the panel; "cw": board at the bottom (see docstring).
VIEWS = {"ep_049": ("L", "ccw"), "ep_050": ("U", "ccw"), "ep_055": ("S", "ccw"), "ep_071": ("C", "cw")}
MOVE_M = 0.005          # the push starts / ends when the box is this far from where it began / finishes
PATH_RGB = (30, 110, 255)
REAL_BLUR = 0.7         # px sigma on the human-video panels, before they are written to a GIF


# -- feasibility --------------------------------------------------------------------------------------------

def check_feasibility(paths_human, ws_map, reach_xy, r_min_cube, pusher_offset, tol):
    """Per path: (n points, fraction feasible, min cube radius from the base). C3.4's test, see
    WorkspaceMap.path_feasibility_fraction: a trajectory is feasible when the fraction is 1."""
    rows = []
    for p in paths_human:
        p = np.asarray(p, float)
        frac = ws_map.path_feasibility_fraction(p, reach_xy, r_min_cube, pusher_offset, tol)
        rows.append((len(p), frac, float(np.linalg.norm(ws_map.transform(p), axis=1).min())))
    return rows


def run_feasibility(episodes, path):
    """Recompute the C3.4 check for the episodes with the stored map and inputs; write the CSV."""
    info = json.loads(WORKSPACE_MAP_PATH.read_text())
    ws = load_workspace_map()
    reach = sample_reachable_xy(n_samples=200_000, seed=42)       # as fit_train_map
    if len(reach) != info["n_fk_samples"]:
        raise SystemExit(f"FK sample has {len(reach)} points at push height, the stored map was fitted on "
                         f"{info['n_fk_samples']}: not the C3.4 sample")
    paths = [np.load(PROCESSED_DIR / f"{e}.npz")["cube_xy_yaw"][:, :2] for e in episodes]
    rows = check_feasibility(paths, ws, reach, info["r_min_cube"], info["pusher_offset"], info["reach_tolerance"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "n_points", "feasible_fraction", "feasible", "min_cube_radius_m", "r_min_cube_m"])
        for e, (n, frac, rmin) in zip(episodes, rows):
            w.writerow([e, n, f"{frac:.4f}", frac >= 1.0, f"{rmin:.4f}", f"{info['r_min_cube']:.4f}"])
    return path


# -- timeline -----------------------------------------------------------------------------------------------

def push_window(xy, t, thresh=MOVE_M):
    """(start, end) times of a push: first time the box is > thresh from its first position, and the time
    after which it stays within thresh of its last. (t[0], t[-1]) if it never moves."""
    xy = np.asarray(xy, float)
    moved = np.flatnonzero(np.linalg.norm(xy - xy[0], axis=1) > thresh)
    away = np.flatnonzero(np.linalg.norm(xy - xy[-1], axis=1) > thresh)
    if not len(moved) or not len(away):
        return float(t[0]), float(t[-1])
    i0, i1 = int(moved[0]), min(int(away[-1]) + 1, len(t) - 1)
    return (float(t[i0]), float(t[i1])) if i1 > i0 else (float(t[0]), float(t[-1]))


def gif_times(real_win, sim_win, lead, tail, fps, max_seconds=None):
    """Seconds relative to the push start for every GIF frame: from -lead to the later push end plus tail."""
    dur = max(real_win[1] - real_win[0], sim_win[1] - sim_win[0]) + tail
    tau = -lead + np.arange(int(round((dur + lead) * fps)) + 1) / fps
    return tau if max_seconds is None else tau[: int(max_seconds * fps) + 1]


# -- real video ---------------------------------------------------------------------------------------------

def detect_track(video_path, step=3):
    """Box marker centres (M, 2) px and the median marker side (px) from every `step`-th frame."""
    from capture.live_check import CUBE_ID, make_detector
    det = make_detector()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    centres, sides, i = [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, frame = cap.retrieve()
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None and CUBE_ID in ids.ravel():
                c = np.asarray(corners[list(ids.ravel()).index(CUBE_ID)]).reshape(4, 2)
                centres.append(c.mean(axis=0))
                sides.append(np.linalg.norm(c[0] - c[1]))
        i += 1
    cap.release()
    return np.array(centres).reshape(-1, 2), (float(np.median(sides)) if sides else np.nan), size


def square_rect(centre, half, size):
    """Integer square (x0, y0, side) of half-width `half` around `centre`, shifted (and shrunk if it must be)
    to lie inside an image of `size` = (width, height)."""
    side = int(min(round(2 * half), size[0], size[1]))
    x0 = int(np.clip(round(centre[0] - side / 2), 0, size[0] - side))
    y0 = int(np.clip(round(centre[1] - side / 2), 0, size[1] - side))
    return x0, y0, side


def real_crop(centres, marker_px, size, props, pad_sides=0.55):
    """Square crop around the box marker's track plus pad_sides box sides; the frame's centre if no detection."""
    if not len(centres) or not np.isfinite(marker_px):
        return square_rect((size[0] / 2, size[1] / 2), min(size) / 2, size)
    box_px = marker_px * props["cube_side"] / props["cube_marker_side"]
    lo, hi = centres.min(axis=0), centres.max(axis=0)
    return square_rect((lo + hi) / 2, np.max(hi - lo) / 2 + pad_sides * box_px, size)


def orient(img, view, native):
    """Rotate an image to the panel orientation. `native` is the image's own orientation: "raw" for the
    recorded frame (x right, y up) or "sim" for the sim's overhead render (x up, y left)."""
    turns = {("raw", "ccw"): cv2.ROTATE_90_COUNTERCLOCKWISE, ("raw", "cw"): cv2.ROTATE_90_CLOCKWISE,
             ("sim", "ccw"): None, ("sim", "cw"): cv2.ROTATE_180}[(native, view)]
    return img if turns is None else cv2.rotate(img, turns)


def real_panels(video_path, csv_path, times, rect, view, size):
    """RGB panels (size x size) of the video at `times` (s from its first frame): the last frame at or before
    each time; times past the end hold the last frame."""
    ts = read_timestamps(csv_path)
    want = np.clip(np.searchsorted(ts, np.asarray(times, float), side="right") - 1, 0, len(ts) - 1)
    x0, y0, s = rect
    cap = cv2.VideoCapture(str(video_path))
    done, i, last = {}, 0, None
    for k in sorted(set(want.tolist())):
        while i < k and cap.grab():
            i += 1
        ok, frame = cap.read()
        i += 1
        if ok:
            crop = frame[y0:y0 + s, x0:x0 + s]
            panel = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
            last = orient(cv2.GaussianBlur(panel, (0, 0), REAL_BLUR), view, "raw")   # sensor noise would bloat the GIF
        if last is None:
            raise SystemExit(f"{video_path}: no frame could be read")
        done[k] = last
    cap.release()
    return [done[k] for k in want]


def plot_letters(results, path, method):
    """One panel per episode in the GIF orientation (letters upright): reference dashed, achieved box path solid."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(3.3 * len(results), 3.9), squeeze=False)
    for ax, r in zip(axes[0], results):
        letter, view = VIEWS.get(r["id"], ("", "ccw"))
        for key, style, label in (("ref_xy", "--", "reference"), ("achieved_xy", "-", "box")):
            p = np.asarray(r[key], float) * 100
            x, y = (-p[:, 1], p[:, 0]) if view == "ccw" else (p[:, 1], -p[:, 0])     # right, up
            ax.plot(x, y, style, lw=1.6, label=label, color="0.55" if key == "ref_xy" else "tab:blue")
            if key == "ref_xy":
                ax.plot(x[0], y[0], "o", color="tab:green", ms=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{letter} ({r['id']})\ndev {r['mean_deviation_cm']:.2f} cm, final {r['final_error_cm']:.2f} cm", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{method}: box path against the reference (start marked), as drawn", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path), dpi=130)
    plt.close(fig)


# -- sim recorder -------------------------------------------------------------------------------------------

class SimRecorder:
    """Wraps a PushTrackEnv and keeps a panel-sized RGB render after reset and after every step.

    Everything else passes through, so eval.report.evaluate/rollout use it as the env. The render is the
    env's "overhead" camera at 960 x 720, cropped to a square around the reference path (+ pad_m), the path
    drawn dashed, then rotated to the view. Uses the env's model and data directly (the env's own render()
    is a fixed 640 x 480 full view).
    """

    def __init__(self, env, size, views, pad_m=0.06, width=960, height=720):
        self.env, self.size, self.views, self.pad_m = env, size, views, pad_m
        self.width, self.height = width, height
        self.frames = {}
        self._renderer = None
        self._cam = mujoco_camera_id(env)

    def __getattr__(self, name):          # only called for names not set on the wrapper
        return getattr(self.env, name)

    def _project(self, xy, z):
        """World (x, y) at height z -> pixel (u, v) in the full render, with the env camera's pose."""
        d = self.env._data
        R, c = d.cam_xmat[self._cam].reshape(3, 3), d.cam_xpos[self._cam]
        p = (np.column_stack([xy, np.full(len(xy), z)]) - c) @ R        # camera frame: x right, y up, -z forward
        f = (self.height / 2) / np.tan(np.radians(self.env._model.cam_fovy[self._cam]) / 2)
        return np.column_stack([self.width / 2 + f * p[:, 0] / -p[:, 2], self.height / 2 - f * p[:, 1] / -p[:, 2]]), f / -p[:, 2].mean()

    def reset(self, *args, **kwargs):
        import mujoco
        out = self.env.reset(*args, **kwargs)
        info = out[1]
        self.ep = info["episode_id"]
        if self._renderer is None:
            m = self.env._model
            m.vis.global_.offwidth, m.vis.global_.offheight = self.width, self.height
            self._renderer = mujoco.Renderer(m, height=self.height, width=self.width)
        self.view = self.views.get(self.ep, ("", "ccw"))[1]
        uv, px_per_m = self._project(self.env.reference_path, self.env._cube_height / 2)
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        self.rect = square_rect((lo + hi) / 2, np.max(hi - lo) / 2 + self.pad_m * px_per_m, (self.width, self.height))
        self.path_px = (uv - self.rect[:2]) * (self.size / self.rect[2])
        self.frames[self.ep] = []
        self._grab()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._grab()
        return out

    def _grab(self):
        self._renderer.update_scene(self.env._data, camera=self._cam)
        x0, y0, s = self.rect
        img = cv2.resize(self._renderer.render()[y0:y0 + s, x0:x0 + s], (self.size, self.size), interpolation=cv2.INTER_AREA)
        pts = np.round(self.path_px).astype(np.int32)
        for i in range(len(pts) - 1):
            if (i // 3) % 2 == 0:                                       # dashes
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), PATH_RGB, 2, cv2.LINE_AA)
        self.frames[self.ep].append(orient(img, self.view, "sim"))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.env.close()


def mujoco_camera_id(env, name="overhead"):
    import mujoco
    cid = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        raise SystemExit(f"the scene has no camera named {name!r}")
    return cid


# -- composing and writing ----------------------------------------------------------------------------------

def label_panel(img, header, footer, header_rgb=(30, 30, 30), footer_rgb=(30, 30, 30)):
    """Panel with a white header and footer band carrying text."""
    h, w = img.shape[:2]
    band = max(18, int(round(h * 0.072)))
    scale = max(0.4, 0.5 * w / 360)
    out = np.full((h + 2 * band, w, 3), 255, np.uint8)
    out[band:band + h] = img
    for text, y, rgb in ((header, band - int(band * 0.3), header_rgb), (footer, h + 2 * band - int(band * 0.3), footer_rgb)):
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, rgb, 1, cv2.LINE_AA)
    return out


def pair_frame(real, sim, tau, tag, method, size):
    """One labelled frame: the human-video panel and the sim panel side by side, panels resized to `size`.
    The timer is under the human video, the legend under the sim."""
    sz = (size, size)
    real = real if real.shape[0] == size else cv2.resize(real, sz, interpolation=cv2.INTER_AREA)
    sim = sim if sim.shape[0] == size else cv2.resize(sim, sz, interpolation=cv2.INTER_AREA)
    left = label_panel(real, "human video", f"{tag}   t = {round(float(tau), 1) + 0.0:.1f} s")
    right = label_panel(sim, f"robot in sim, {method}", "-- reference path", footer_rgb=PATH_RGB)
    return np.concatenate([left, right], axis=1)


def save_gif(frames, path, fps, max_mb=8.0, ladder=((1, 256), (1, 128), (2, 256), (2, 128), (2, 64), (3, 64), (3, 32))):
    """Write a looping GIF with one palette for all frames (no dithering, so flat areas stay flat and the
    frame-to-frame differences stay small). Steps down through `ladder` = (keep every n-th frame, colours)
    until the file is under max_mb. Returns (MB, stride, colours)."""
    from PIL import Image
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for stride, colours in ladder:
        use = frames[::stride]
        sample = np.concatenate([use[i] for i in np.linspace(0, len(use) - 1, min(8, len(use))).astype(int)], axis=0)
        pal = Image.fromarray(sample).quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in use]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps * stride)), loop=0)
        mb = Path(path).stat().st_size / 1e6
        if mb <= max_mb:
            break
    else:
        print(f"warning: {path} is {mb:.1f} MB, over {max_mb} MB at the last setting of the ladder")
    return mb, stride, colours


def grid(cells, cols=2):
    """Tile equal-sized frames into rows of `cols`."""
    rows = [np.concatenate(cells[i:i + cols], axis=1) for i in range(0, len(cells), cols)]
    return np.concatenate(rows, axis=0)


# -- main ---------------------------------------------------------------------------------------------------

def load_controller(args):
    """(controller, method, k_ref, max_episode_steps) from --controller scripted or --run."""
    if args.run:
        import torch
        torch.set_num_threads(1)
        from rl.eval_policy import load_controller as load_run
        cfg = json.loads((Path(args.run) / "config.json").read_text())
        k_ref = cfg.get("k_ref", 5)
        ctrl = load_run(args.run, args.model, k_ref, cfg.get("residual_scale"))
        return ctrl, args.method or Path(args.run).name, k_ref, args.max_steps or cfg.get("max_episode_steps", 600)
    from control.scripted_pusher import ScriptedPusher
    return ScriptedPusher(), args.method or "scripted", 5, args.max_steps or 600


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=("scripted",))
    ap.add_argument("--run", help="a trained run directory (runs/<name>), loaded through rl.eval_policy")
    ap.add_argument("--model", help="checkpoint .zip of --run instead of its final model.zip")
    ap.add_argument("--method", help="label for files and the sim panel (default: scripted, or the run's name)")
    ap.add_argument("--episodes", nargs="*", help="showcase episodes to run (default: all)")
    ap.add_argument("--max-steps", type=int, help="env step limit (default 600, or the run's)")
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--lead", type=float, default=1.0, help="seconds shown before the push starts")
    ap.add_argument("--tail", type=float, default=1.5, help="seconds held after the later push ends")
    ap.add_argument("--panel", type=int, default=360, help="panel side in the per-episode GIFs (px)")
    ap.add_argument("--grid-panel", type=int, default=200, help="panel side in the combined 2x2 GIF (px)")
    ap.add_argument("--max-mb", type=float, default=8.0, help="size limit per GIF; colours and frame rate are cut to fit")
    ap.add_argument("--max-seconds", type=float, help="cap each GIF at this many seconds (tests)")
    ap.add_argument("--no-gif", action="store_true", help="metrics, CSV and overlay only")
    ap.add_argument("--recheck-feasibility", action="store_true")
    ap.add_argument("--skip-feasibility", action="store_true")
    args = ap.parse_args(argv)
    if bool(args.controller) == bool(args.run):
        ap.error("give exactly one of --controller scripted and --run <runs dir>")

    splits = json.loads(SPLITS_PATH.read_text())
    episodes = args.episodes or splits["showcase"]
    for e in episodes:
        if e not in splits["showcase"]:
            ap.error(f"{e} is not a showcase episode")
    out = args.out_dir

    # 1. feasibility under the frozen map
    feas = out / "showcase_feasibility.csv"
    if not args.skip_feasibility and (args.recheck_feasibility or not feas.exists()):
        t0 = time.time()
        run_feasibility(splits["showcase"], feas)
        print(f"feasibility: {feas} ({time.time() - t0:.0f} s)")
    if feas.exists():
        print(feas.read_text())

    # 2. the controller on the showcase episodes, evaluation protocol
    ctrl, method, k_ref, steps = load_controller(args)
    env = PushTrackEnv(split="showcase", episodes=episodes, max_episode_steps=steps, k_ref=k_ref)
    if hasattr(ctrl, "max_delta"):
        ctrl.max_delta = env.max_delta
    rec = SimRecorder(env, args.panel, VIEWS) if not args.no_gif else env
    t0 = time.time()
    results = evaluate(ctrl, env=rec, method=method, until_done=True)
    print(f"{len(results)} episodes in {time.time() - t0:.0f} s")
    text = category_table(results) + "\n\n" + "\n".join(
        f"{r['id']}: success={bool(r['success'])} progress={r['progress']:.2f} dev={r['mean_deviation_cm']:.2f} cm "
        f"final={r['final_error_cm']:.2f} cm time={r['completion_time']:.1f} s steps={r['steps']}" for r in results)
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    write_results_csv(results, out / f"showcase_{method}.csv")
    plot_overlay(results, out / f"showcase_{method}_overlay.png",
                 title=f"{method}, showcase letters: reference (dashed) vs achieved box path")
    plot_letters(results, out / f"showcase_{method}_letters.png", method)
    (out / f"showcase_{method}_summary.txt").write_text(text + "\n")
    if args.no_gif:
        env.close()
        return results

    # 3. GIFs
    props = load_props(require=("cube_side", "cube_marker_side"))
    dt = env.dt
    per_ep = []
    for r in results:
        ep = r["id"]
        letter = VIEWS.get(ep, ("", "ccw"))
        d = np.load(PROCESSED_DIR / f"{ep}.npz")
        real_win = push_window(d["cube_xy_yaw"][:, :2], d["t"])
        sim_t = np.arange(len(r["achieved_xy"])) * dt
        sim_win = push_window(r["achieved_xy"], sim_t)
        taus = gif_times(real_win, sim_win, args.lead, args.tail, args.fps, args.max_seconds)
        video, tcsv = args.raw_dir / f"{ep}.avi", args.raw_dir / f"{ep}_t.csv"
        centres, marker_px, vsize = detect_track(video)
        if not len(centres):
            print(f"{ep}: no box marker detected, cropping the frame centre")
        rect = real_crop(centres, marker_px, vsize, props)
        real = real_panels(video, tcsv, real_win[0] + taus, rect, letter[1], args.panel)
        frames = rec.frames.pop(ep)
        sim = [frames[int(np.clip(round((sim_win[0] + tau) / dt), 0, len(frames) - 1))] for tau in taus]
        tag = f"{letter[0]} ({ep})" if letter[0] else ep
        path = out / f"showcase_{method}_{ep}.gif"
        mb, stride, colours = save_gif([pair_frame(a, b, tau, tag, method, args.panel) for a, b, tau in zip(real, sim, taus)],
                                       path, args.fps, args.max_mb)
        print(f"{ep}: real push {real_win[0]:.1f}-{real_win[1]:.1f} s, sim push {sim_win[0]:.1f}-{sim_win[1]:.1f} s, "
              f"{len(taus)} frames, {path} {mb:.1f} MB (every {stride} frame, {colours} colours)")
        per_ep.append([pair_frame(a, b, tau, letter[0] or ep, method, args.grid_panel) for a, b, tau in zip(real, sim, taus)])
        del real, sim, frames
    env.close()

    n = max(len(f) for f in per_ep)
    combined = [grid([f[min(k, len(f) - 1)] for f in per_ep]) for k in range(n)]
    path = out / f"showcase_{method}.gif"
    mb, stride, colours = save_gif(combined, path, args.fps, args.max_mb)
    print(f"combined: {len(combined)} frames {combined[0].shape[1]}x{combined[0].shape[0]}, {path} {mb:.1f} MB "
          f"(every {stride} frame, {colours} colours)")
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
