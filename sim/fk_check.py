"""Check the PoE forward kinematics against the MuJoCo SO-101 model (G1 / C3.3).

    python -m sim.fk_check [--n 1000] [--seed 0]

The PoE model is built from the URDF (control/kinematics), MuJoCo's from the MJCF (sim/assets/so101).
Both come from the same CAD export, so this tests the parser, the screw axes, the joint zeros and signs, and
the end-effector frame. Four checks:

1. Screw axes at q = 0: MuJoCo's world-frame joint axes and anchors give a screw [w; -w x p] for each joint,
   compared column by column with the PoE Slist. This is where a joint-zero or sign mismatch would show.
2. Home pose: MuJoCo's end-effector site against M.
3. Random configurations uniform within the joint limits: PoE FK position against the MuJoCo site position.
4. IK in MuJoCo: position-only DLS IK (as the controller will use it) toward positions taken from MuJoCo, then
   the solution is set in MuJoCo and where its site actually lands is measured.

The MJCF `gripperframe` site is not oriented like the URDF's `gripper_frame_link` (90 vs 180 degrees about y),
so orientation is reported as a *constant frame offset* (its spread over configurations should be ~0),
not as an error. The arm is driven by position-only IK, so position is the quantity that matters.
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from control.kinematics.core import Adjoint, FKinBody, FKinSpace, TransInv
from control.kinematics.ik import IKinBodyDLS
from control.kinematics.parser import DEFAULT_URDF, findMnS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MJCF = REPO_ROOT / "sim" / "assets" / "so101" / "so101_new_calib.xml"
EE_SITE = "gripperframe"      # MJCF counterpart of the URDF's gripper_frame_link
RESULTS_DIR = REPO_ROOT / "results"


def arm_joint_names(urdf_path=DEFAULT_URDF):
    """Joint names in the order findMnS builds its Slist columns (reversed URDF order, arm joints only)."""
    root = ET.parse(urdf_path).getroot()
    return [j.get("name") for j in reversed(root.findall("joint"))
            if j.get("type") != "fixed" and j.get("name") != "gripper" and j.find("origin") is not None]


def rot_angle(R):
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def load(mjcf=DEFAULT_MJCF):
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    return m, mujoco.MjData(m)


def set_arm(m, d, names, q):
    d.qpos[:] = 0
    for name, v in zip(names, q):
        d.qpos[m.joint(name).qposadr[0]] = v
    mujoco.mj_kinematics(m, d)


def ee_pose(m, d):
    sid = m.site(EE_SITE).id
    T = np.eye(4)
    T[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = d.site_xpos[sid]
    return T


def screw_axes_at_zero(m, d, names):
    """Space-frame screws [w; -w x p] from MuJoCo's own joint axes and anchors at q = 0."""
    set_arm(m, d, names, np.zeros(len(names)))
    S = np.zeros((6, len(names)))
    for i, name in enumerate(names):
        j = m.joint(name).id
        w, p = d.xaxis[j].copy(), d.xanchor[j].copy()
        S[:3, i], S[3:, i] = w, -np.cross(w, p)
    return S


def run(n=1000, seed=0, mjcf=DEFAULT_MJCF, urdf=DEFAULT_URDF):
    """Returns a dict of results; see main() for what is printed."""
    M, Slist, limits = findMnS(urdf)
    names = arm_joint_names(urdf)
    m, d = load(mjcf)
    assert len(names) == Slist.shape[1], (names, Slist.shape)

    # joint limits must agree, or random configs would mean different things in the two models
    mj_limits = np.array([m.jnt_range[m.joint(nm).id] for nm in names])
    out = {"names": names, "limit_diff": float(np.abs(mj_limits - limits).max())}

    S_mj = screw_axes_at_zero(m, d, names)
    out["screw_diff"] = float(np.abs(S_mj - Slist).max())
    out["screw_diff_per_joint"] = np.abs(S_mj - Slist).max(axis=0)

    T_home = ee_pose(m, d)          # d is at q = 0 after screw_axes_at_zero
    out["home_pos_err"] = float(np.linalg.norm(T_home[:3, 3] - M[:3, 3]))
    out["home_frame_offset_deg"] = float(np.degrees(rot_angle(M[:3, :3].T @ T_home[:3, :3])))

    rng = np.random.default_rng(seed)
    q = rng.uniform(limits[:, 0], limits[:, 1], size=(n, len(names)))
    pos_err, offsets = np.empty(n), []
    for k in range(n):
        set_arm(m, d, names, q[k])
        T_mj = ee_pose(m, d)
        T_poe = FKinSpace(M, Slist, q[k])
        pos_err[k] = np.linalg.norm(T_poe[:3, 3] - T_mj[:3, 3])
        offsets.append(T_poe[:3, :3].T @ T_mj[:3, :3])
    R_ref = offsets[0]
    spread = np.degrees([rot_angle(R_ref.T @ R) for R in offsets])
    out.update(q=q, pos_err=pos_err, offset_spread_deg=np.array(spread),
               offset_deg=float(np.degrees(rot_angle(R_ref))), n=n, seed=seed)
    return out


def ik_in_mujoco(n=200, seed=0, noise=0.3, ev=1e-3, mjcf=DEFAULT_MJCF, urdf=DEFAULT_URDF):
    """Position-only DLS IK toward MuJoCo-derived targets, scored by MuJoCo's own end-effector position.

    The target is the MuJoCo site position at a random configuration; the IK starts ``noise`` rad away from it
    (the controller warm-starts from the previous solution, so starts are near). The target orientation is the
    start configuration's own, so the unconstrained orientation does not fight the position task; with one
    fixed orientation (the home pose) only ~60% of these solves converged.

    ``miss`` counts solutions the solver reported as converged whose true position error exceeds ``ev``.
    IKinBodyDLS clamps to the joint limits after it evaluates the error, so a solution that overshoots a limit is
    returned clamped but verified unclamped; ``miss_at_limit`` is how many misses have a joint on a limit.
    """
    M, Slist, limits = findMnS(urdf)
    Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(Slist.shape[1])]).T
    names = arm_joint_names(urdf)
    m, d = load(mjcf)
    rng = np.random.default_rng(seed)
    errs, ok, miss, miss_at_limit = [], 0, 0, 0
    for _ in range(n):
        q_true = rng.uniform(limits[:, 0], limits[:, 1])
        set_arm(m, d, names, q_true)
        target = ee_pose(m, d)[:3, 3]
        q0 = np.clip(q_true + rng.uniform(-noise, noise, q_true.shape), limits[:, 0], limits[:, 1])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = FKinBody(M, Blist, q0)[:3, :3], target
        q, success = IKinBodyDLS(Blist, M, T, q0, limits, ev=ev, position_only=True)
        if success:
            ok += 1
            set_arm(m, d, names, q)
            err = np.linalg.norm(ee_pose(m, d)[:3, 3] - target)
            errs.append(err)
            if err > ev:
                miss += 1
                miss_at_limit += bool(np.any(np.isclose(q, limits[:, 0], atol=1e-12) | np.isclose(q, limits[:, 1], atol=1e-12)))
    return {"n": n, "success_rate": ok / n, "errs": np.array(errs), "noise": noise, "ev": ev,
            "miss": miss, "miss_at_limit": miss_at_limit}


def write_tables(res, out_dir=RESULTS_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pe = res["pos_err"] * 1000
    with open(out_dir / "fk_check_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_configs", "seed", "pos_err_mean_mm", "pos_err_median_mm", "pos_err_p99_mm", "pos_err_max_mm",
                    "screw_axis_max_diff", "frame_offset_deg", "frame_offset_spread_max_deg",
                    "ik_n", "ik_success_rate", "ik_mujoco_err_mean_mm", "ik_mujoco_err_max_mm", "ik_miss", "ik_miss_at_limit"])
        w.writerow([res["n"], res["seed"], f"{pe.mean():.6f}", f"{np.median(pe):.6f}", f"{np.percentile(pe, 99):.6f}",
                    f"{pe.max():.6f}", f"{res['screw_diff']:.3e}", f"{res['offset_deg']:.3f}",
                    f"{res['offset_spread_deg'].max():.6f}", res["ik"]["n"], f"{res['ik']['success_rate']:.3f}",
                    f"{res['ik']['errs'].mean() * 1000:.4f}", f"{res['ik']['errs'].max() * 1000:.4f}",
                    res["ik"]["miss"], res["ik"]["miss_at_limit"]])
    with open(out_dir / "fk_check_configs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q_{nm}_rad" for nm in res["names"]] + ["pos_err_mm"])
        for q, e in zip(res["q"], pe):
            w.writerow([f"{v:.6f}" for v in q] + [f"{e:.6f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="random configurations (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--no-write", action="store_true", help=f"do not write tables to {RESULTS_DIR}")
    args = ap.parse_args(argv)
    res = run(args.n, args.seed, args.mjcf)
    res["ik"] = ik_in_mujoco(seed=args.seed, mjcf=args.mjcf)
    pe = res["pos_err"] * 1000
    print(f"joints (PoE column order): {res['names']}")
    print(f"joint limits, MJCF vs URDF: max difference {res['limit_diff']:.2e} rad")
    print(f"screw axes at q=0, MuJoCo vs PoE Slist: max difference {res['screw_diff']:.2e} "
          f"(per joint {np.array2string(res['screw_diff_per_joint'], precision=1)})")
    print(f"home pose: position difference {res['home_pos_err'] * 1000:.4f} mm; "
          f"frame offset {res['home_frame_offset_deg']:.2f} deg (MJCF site vs URDF frame)")
    print(f"\nFK position error over {res['n']} random configurations (seed {res['seed']}):")
    print(f"  max    {pe.max():.5f} mm\n  mean   {pe.mean():.5f} mm\n  median {np.median(pe):.5f} mm\n"
          f"  p99    {np.percentile(pe, 99):.5f} mm")
    print(f"frame offset between the two EE frames: {res['offset_deg']:.2f} deg, spread over configs "
          f"{res['offset_spread_deg'].max():.2e} deg (should be ~0: constant offset)")
    ik = res["ik"]
    print(f"\nIK in MuJoCo (position-only DLS, ev={ik['ev'] * 1000:.1f} mm, start {ik['noise']} rad off): "
          f"{ik['success_rate'] * 100:.1f}% converged of {ik['n']}; MuJoCo end-effector error of the solutions "
          f"mean {ik['errs'].mean() * 1000:.4f} mm, max {ik['errs'].max() * 1000:.4f} mm")
    print(f"  converged but position error > ev: {ik['miss']} (of which a joint is on its limit: {ik['miss_at_limit']})"
          f" -> check FK on returned angles, the solver clamps after it evaluates the error")
    if not args.no_write:
        write_tables(res)
        print(f"tables written to {RESULTS_DIR}")
    return res


if __name__ == "__main__":
    main()
