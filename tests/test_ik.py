"""Inverse-kinematics verification: round-trip success, convergence radius, and
singularity behaviour.

Deterministic and headless by design: this is the oracle the V1 operator-extraction
refactor is verified against, so an unseeded or blocking check is worthless.

Run from the repo root:
    python -m tests.test_ik              # headless checks + threshold assertions
    python -m tests.test_ik --plots      # also write figures to docs/media/
    python -m tests.test_ik --capture    # write the bit-exact refactor baseline
    python -m tests.test_ik --check      # compare solutions against that baseline

TWO STANDARDS, DELIBERATELY SPLIT (see the V0 notes in PLAN.md):
  - The committed assertions are success-rate THRESHOLDS. Portable: different
    BLAS builds can flip borderline convergences, so exact rates are not a
    cross-machine invariant.
  - `--capture` / `--check` is a stricter, LOCAL, single-use gate for the refactor
    moment: the solved joint angles must come back bit-identical (max|dtheta| == 0),
    because moving operations into a function without reordering them should not
    perturb the arithmetic. A tiny nonzero delta (~1e-16) is not noise to wave
    through -- it means something got reassociated, and that should be explained
    before it is accepted. The baseline file is gitignored and is not meant to
    outlive the commit it gates.

GATE ALIGNMENT (why the reported rates jumped ~43 points at V0). The scoring gate
used to demand a linear error < 1e-3 while the solver was only ever asked for
ev=5e-3, so legitimate convergences were scored as failures: at noise=1, 44.6% of
solver-converged solves land in [1e-3, 5e-3) and only 1.6% genuinely exceed 5e-3.
That single mismatch is the whole reason this test reported ~52% where DEVLOG
Entry 6's sweep recorded 95.0%. The gate now reuses the SAME tolerances handed to
the solver, which reproduces Entry 6's table across all ten noise levels. Keep
them tied -- they are one decision, not two.
"""
import argparse
from pathlib import Path

import numpy as np

from control.kinematics.core import FKinBody, JacobianBody, TransInv, MatrixLog6, se3ToVec
from control.kinematics.ik import IKinBodyDLS
from tests.robot import M, Blist, limits

# --- determinism ----------------------------------------------------------
# One base seed; each noise level offsets it so the levels don't all replay the
# same theta samples. np.random.default_rng's stream is stability-guaranteed by
# NumPy's policy, so these results are reproducible across versions.
SEED = 20260825
ITERS = 200

# Solver tolerances. Passed to IKinBodyDLS *and* used to score its output -- see
# the GATE ALIGNMENT note above.
SOLVER_EOMG = 1e-2      # rad
SOLVER_EV = 5e-3        # m

NOISE_LEVELS = [0.0, 0.01, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 5.0]

# Minimum acceptable success rate (%) per noise level. Set a few points below the
# seeded measurement so a BLAS difference doesn't fail the suite, but tight enough
# to catch a real regression. Reference values are DEVLOG Entry 6's table.
MIN_SUCCESS = {
    0.0: 99.0, 0.01: 99.0, 0.1: 97.0, 0.5: 93.0, 0.75: 93.0,
    1.0: 85.0, 1.25: 85.0, 1.5: 83.0, 2.0: 80.0, 5.0: 66.0,
}

# DEVLOG Entry 6's measured sweep, for side-by-side reporting (not asserted).
ENTRY6 = {
    0.0: 100.0, 0.01: 100.0, 0.1: 100.0, 0.5: 98.5, 0.75: 96.5,
    1.0: 95.0, 1.25: 90.0, 1.5: 85.5, 2.0: 83.0, 5.0: 72.5,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / ".ik_baseline.npz"
# Regenerable test output, kept out of docs/media/ proper: that directory is
# curated for README/DEVLOG, and mixing it with figures anyone can re-emit via
# --plots means git status noise on every run. Gitignored.
MEDIA_DIR = REPO_ROOT / "docs" / "media" / "tests"


def round_trip(joint_limits, iters, noise, *, seed, eomg=SOLVER_EOMG, ev=SOLVER_EV):
    """FK(theta) -> IK -> compare. Returns (success_rate_pct, solutions).

    Sample a random valid theta, FK it to a target, perturb theta by `noise` to
    make the initial guess, solve, and confirm the solution reproduces the target
    within the same tolerances the solver was asked for. Non-converged solves
    count as failures.

    `solutions` is the (iters, n) array of solved joint angles -- returned
    unconditionally so --capture/--check can compare them bit-for-bit.
    """
    rng = np.random.default_rng(seed)
    solutions = np.empty((iters, joint_limits.shape[0]))
    passed = 0

    for i in range(iters):
        thetalist = rng.uniform(low=joint_limits[:, 0], high=joint_limits[:, 1])
        T_target = FKinBody(M, Blist, thetalist)
        theta_init = thetalist + rng.uniform(-noise, noise, size=thetalist.shape)
        theta_solved, success = IKinBodyDLS(Blist, M, T_target, theta_init,
                                            joint_limits, eomg=eomg, ev=ev)
        solutions[i] = theta_solved

        if not success:
            continue
        Vb = se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, theta_solved)) @ T_target))
        if np.linalg.norm(Vb[:3]) < eomg and np.linalg.norm(Vb[3:]) < ev:
            passed += 1

    return 100.0 * passed / iters, solutions


def noise_sweep(iters=ITERS):
    """Round-trip success across initial-guess noise -- the convergence-radius
    characterisation behind Entry 6's in-loop-clamping decision. Returns
    {noise: (rate_pct, solutions)}. No plotting; see --plots."""
    return {
        noise: round_trip(limits, iters, noise, seed=SEED + i)
        for i, noise in enumerate(NOISE_LEVELS)
    }


def verify_singular(maxiters=50, lam=0.05):
    """Steps DLS against a plain pseudoinverse near an elbow singularity.

    Note this deliberately reimplements the step inline rather than calling
    IKinBodyDLS: it compares two *stepping rules*, so it must not inherit the
    solver's choice of one. That also makes it independent of the V1 refactor.

    Tolerances here are deliberately NOT SOLVER_EV. The gate-alignment argument
    applies to scoring the solver against what it was asked for; nothing is being
    scored against the solver here. The tighter 1e-3 linear gate is kept because
    it costs a second iteration and two points of comparison discriminate the two
    stepping rules better than one -- and because it reproduces the iteration
    counts DEVLOG Entry 6 records.

    Returns (cond, hist_dls, hist_pinv) -- caller prints/plots.
    """
    eomg, ev = 1e-2, 1e-3
    theta_singular = np.array([0.0, np.pi / 2, 0.0, 0.0, 0.0])
    theta_init = theta_singular.copy()
    T_target = FKinBody(M, Blist, theta_singular + np.array([0.0, 0.1, 0.0, 0.1, 0.0]))
    cond = float(np.linalg.cond(JacobianBody(Blist, theta_singular)))

    def run_ik(use_dls):
        theta = theta_init.copy()
        history = [theta.copy()]
        for _ in range(maxiters):
            Tsb = FKinBody(M, Blist, theta)
            Vb = se3ToVec(MatrixLog6(TransInv(Tsb) @ T_target))
            if np.linalg.norm(Vb[:3]) < eomg and np.linalg.norm(Vb[3:]) < ev:
                break
            J = JacobianBody(Blist, theta)
            if use_dls:
                delta = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), Vb)
            else:
                delta = np.linalg.pinv(J) @ Vb
            if not np.all(np.isfinite(delta)):
                break
            theta = theta + delta
            history.append(theta.copy())
        return np.array(history)

    return cond, run_ik(use_dls=True), run_ik(use_dls=False)


# --- bit-exact refactor gate ----------------------------------------------
def _key(noise):
    return f"noise_{noise}"


def capture_baseline(results, path=BASELINE_PATH):
    np.savez(path, **{_key(n): sols for n, (_, sols) in results.items()})
    total = sum(sols.size for _, sols in results.values())
    print(f"\nbaseline written: {path}  ({total} joint values across "
          f"{len(results)} noise levels)")
    print("gitignored and single-use -- delete it once V1 is verified.")


def check_baseline(results, path=BASELINE_PATH):
    """Compare solved angles against the captured baseline. Returns True on
    bit-exact match. Anything nonzero is reported, not tolerated."""
    if not path.exists():
        raise SystemExit(f"no baseline at {path} -- run --capture on the "
                         f"pre-refactor code first.")
    ref = np.load(path)
    print(f"\nbit-exact check against {path.name}")
    worst = 0.0
    for noise, (_, sols) in results.items():
        if _key(noise) not in ref:
            raise SystemExit(f"baseline is missing {_key(noise)} -- recapture.")
        # Guard the --iters footgun: mismatched trial counts would otherwise
        # broadcast into a meaningless comparison instead of failing loudly.
        if ref[_key(noise)].shape != sols.shape:
            raise SystemExit(
                f"baseline shape {ref[_key(noise)].shape} != current {sols.shape} "
                f"at noise={noise} -- capture and check must use the same --iters.")
        delta = float(np.max(np.abs(sols - ref[_key(noise)])))
        worst = max(worst, delta)
        flag = "exact" if delta == 0.0 else f"DIFFERS  max|dtheta|={delta:.3e}"
        print(f"  noise={noise:<5} {flag}")

    if worst == 0.0:
        print("\nPASS  all solutions bit-identical -- refactor changed no arithmetic.")
        return True
    print(f"\nFAIL  max|dtheta| = {worst:.3e} across all levels.")
    print("Not a tolerance question: identify which operation was reassociated "
          "before accepting this.")
    return False


# --- optional figures ------------------------------------------------------
def save_plots(results, singular):
    """Write the two characterisation figures. Imported lazily and forced onto a
    non-interactive backend so the default path never needs matplotlib or a display."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(NOISE_LEVELS, [results[n][0] for n in NOISE_LEVELS], marker="o")
    plt.xscale("symlog", linthresh=0.01)
    plt.xlabel("Noise magnitude (rad)")
    plt.ylabel("Success rate (%)")
    plt.title("IK round-trip success vs initial guess noise")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.tight_layout()
    sweep_path = MEDIA_DIR / "ik_noise_sweep.png"
    plt.savefig(sweep_path, dpi=150)
    plt.close()

    cond, hist_dls, hist_pinv = singular
    labels = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, label in enumerate(labels):
        axes[0].plot(hist_dls[:, i], label=label)
        axes[1].plot(hist_pinv[:, i], label=label)
    axes[0].set_title("DLS (lam=0.05)")
    axes[1].set_title("Pseudoinverse")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Joint angle (rad)")
        ax.legend(fontsize=8)
        ax.grid(True)
    plt.suptitle(f"IK near elbow singularity (shoulder_lift=pi/2), cond={cond:.1f}")
    plt.tight_layout()
    sing_path = MEDIA_DIR / "ik_singularity.png"
    plt.savefig(sing_path, dpi=150)
    plt.close()

    print(f"\nfigures written:\n  {sweep_path}\n  {sing_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true", help="write figures to docs/media/")
    parser.add_argument("--capture", action="store_true", help="write the bit-exact baseline")
    parser.add_argument("--check", action="store_true", help="compare against the baseline")
    parser.add_argument("--iters", type=int, default=ITERS, help=f"trials per noise level (default {ITERS})")
    args = parser.parse_args()

    print(f"seed={SEED}  iters={args.iters}  gate: eomg={SOLVER_EOMG} ev={SOLVER_EV} "
          f"(same tolerances handed to the solver)\n")

    results = noise_sweep(iters=args.iters)

    print(f"{'noise':>6} | {'success':>8} | {'min':>6} | {'Entry6':>7} | result")
    print("-" * 52)
    failures = []
    for noise in NOISE_LEVELS:
        rate = results[noise][0]
        floor = MIN_SUCCESS[noise]
        ok = rate >= floor
        if not ok:
            failures.append((noise, rate, floor))
        print(f"{noise:6.2f} | {rate:7.1f}% | {floor:5.1f}% | {ENTRY6[noise]:6.1f}% | "
              f"{'PASS' if ok else 'FAIL'}")

    singular = verify_singular()
    cond, hist_dls, hist_pinv = singular
    print(f"\nsingularity (shoulder_lift=pi/2): cond={cond:.1f}  "
          f"DLS {len(hist_dls) - 1} iters  pinv {len(hist_pinv) - 1} iters")

    if args.capture:
        capture_baseline(results)
    if args.check and not check_baseline(results):
        failures.append(("bit-exact", 0.0, 0.0))
    if args.plots:
        save_plots(results, singular)

    if failures:
        raise SystemExit(f"\nFAILED: {len(failures)} check(s) below threshold.")
    print("\nOK  all noise levels above threshold.")


if __name__ == "__main__":
    main()
