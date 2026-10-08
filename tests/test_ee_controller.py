"""Tests for C3.3: EE velocity controller (DLS IK wrapper).

Key regression test: the wrapper must reject a "converged" IK solution that
hit a joint limit and misses the position tolerance.

Decision log 2026-10-07: IKinBodyDLS clamps joints after the error check, so
a solution that overshot a limit is returned clamped but verified unclamped.
In the G1 sweep (200 targets, noise 0.3 rad) 8 solutions met the convergence
criterion yet missed by > 1 mm; all 8 had a joint exactly on a limit.
The EEController wrapper catches this by re-running FK on the returned angles.

Also reports env steps/sec with the controller in the loop (RL dependency).

Run:  python -m pytest tests/test_ee_controller.py -v
"""
import time

import numpy as np
import pytest

from control.ee_controller import EEController
from control.kinematics.core import Adjoint, FKinBody, TransInv
from control.kinematics.ik import IKinBodyDLS
from control.kinematics.parser import DEFAULT_URDF, findMnS


def _build_blist():
    M, Slist, limits = findMnS(DEFAULT_URDF)
    n = Slist.shape[1]
    Blist = np.array([Adjoint(TransInv(M)) @ Slist[:, i] for i in range(n)]).T
    return M, Blist, limits


def test_ik_rejects_limit_miss():
    """EEController must return success=False whenever raw IK would produce a
    limit-clamped miss.  Zero false-accepts expected.

    The test sweeps 150 (q0, target) pairs with noise=0.3 rad (matching the G1
    sweep in fk_check.ik_in_mujoco).  It first confirms raw IKinBodyDLS produces
    at least one converged-but-wrong solution in this sweep (validates the test
    setup), then checks the controller produces zero false-accepts.
    """
    M, Blist, limits = _build_blist()
    ev = 1e-3
    noise = 0.3
    n = 150
    seed = 3  # seed that produces misses in the G1 sweep

    ctrl = EEController(ev=ev)
    rng  = np.random.default_rng(seed)

    raw_misses = 0
    ctrl_false_accepts = 0

    for _ in range(n):
        q_true = rng.uniform(limits[:, 0], limits[:, 1])
        T_true = FKinBody(M, Blist, q_true)
        target = T_true[:3, 3].copy()
        q0 = np.clip(q_true + rng.uniform(-noise, noise, q_true.shape),
                     limits[:, 0], limits[:, 1])

        # --- raw IK (no re-check) ---
        T_tgt = FKinBody(M, Blist, q0).copy()
        T_tgt[:3, 3] = target
        q_raw, converged = IKinBodyDLS(
            Blist, M, T_tgt, q0, limits, ev=ev, position_only=True
        )
        if converged:
            err = np.linalg.norm(FKinBody(M, Blist, q_raw)[:3, 3] - target)
            if err > ev:
                raw_misses += 1

        # --- EEController (with re-check) ---
        ctrl.reset(q0=q0)
        q_ctrl, success = ctrl.solve(target[:2], z_fixed=float(target[2]))
        if success:
            err = np.linalg.norm(FKinBody(M, Blist, q_ctrl)[:3, 3] - target)
            if err > ev:
                ctrl_false_accepts += 1

    assert raw_misses > 0, (
        "No converged-but-wrong solutions found in this sweep; "
        "the test is not exercising the joint-limit clamp scenario"
    )
    assert ctrl_false_accepts == 0, (
        f"EEController accepted {ctrl_false_accepts} solutions that miss by > {ev * 1000:.0f} mm"
    )


def test_orientation_from_current_fk():
    """The target orientation is always the start FK rotation, never the home pose.

    Without this, convergence drops from ~100 % to ~61 % (decision log 2026-10-07).
    We check that a high-noise sweep still converges > 90 % of the time.
    """
    M, Blist, limits = _build_blist()
    ctrl = EEController(ev=1e-3)
    rng  = np.random.default_rng(7)
    n, noise = 80, 0.3

    successes = 0
    for _ in range(n):
        q_true = rng.uniform(limits[:, 0], limits[:, 1])
        target = FKinBody(M, Blist, q_true)[:3, 3].copy()
        q0 = np.clip(q_true + rng.uniform(-noise, noise, q_true.shape),
                     limits[:, 0], limits[:, 1])
        ctrl.reset(q0=q0)
        _, ok = ctrl.solve(target[:2], z_fixed=float(target[2]))
        if ok:
            successes += 1

    rate = successes / n
    assert rate > 0.90, f"IK convergence {rate * 100:.0f}% < 90%; orientation from FK may be broken"


def test_step_integrates_velocity():
    """step(vx, vy, dt) moves the internal EE position by approximately (vx*dt, vy*dt)."""
    M, Blist, limits = _build_blist()
    ctrl = EEController(ev=1e-3)

    # Start from a configuration that is close to the workspace centre
    rng = np.random.default_rng(1)
    q0 = rng.uniform(limits[:, 0], limits[:, 1])
    ctrl.reset(q0=q0)
    T0 = ctrl.ee_pose()

    vx, vy, dt = 0.05, -0.03, 0.05
    q1, ok = ctrl.step(vx, vy, dt)
    if not ok:
        pytest.skip("IK did not converge from this start; skip velocity test")

    T1 = ctrl.ee_pose()
    dx = T1[0, 3] - T0[0, 3]
    dy = T1[1, 3] - T0[1, 3]

    # The achieved displacement should be close to vx*dt, vy*dt (within 1 mm)
    assert abs(dx - vx * dt) < 1e-3, f"dx error {abs(dx - vx * dt) * 1000:.2f} mm"
    assert abs(dy - vy * dt) < 1e-3, f"dy error {abs(dy - vy * dt) * 1000:.2f} mm"


def test_controller_throughput():
    """Report env steps/sec with the IK controller in the loop.

    Not a threshold test: just ensures the controller runs and prints the rate
    so the RL setup can check feasibility.  Typical target: > 500 solves/sec.
    """
    ctrl = EEController(ev=1e-3)
    M, Blist, limits = _build_blist()
    rng = np.random.default_rng(5)

    # Build 200 targets near the workspace
    q_starts  = rng.uniform(limits[:, 0], limits[:, 1], size=(200, limits.shape[0]))
    targets   = np.array([
        FKinBody(M, Blist, q)[:3, 3]
        for q in q_starts
    ])

    n_reps = 5
    t0 = time.perf_counter()
    for _ in range(n_reps):
        for q0, tgt in zip(q_starts, targets):
            ctrl.reset(q0=q0)
            ctrl.solve(tgt[:2], z_fixed=float(tgt[2]))
    elapsed = time.perf_counter() - t0
    rate = n_reps * len(q_starts) / elapsed

    print(f"\nEEController throughput: {rate:.0f} solves/sec ({elapsed:.2f} s for {n_reps * len(q_starts)} solves)")
    assert rate > 0, "throughput must be positive (sanity)"


# -- warm-started chunked solve (env throughput) ---------------------------------------------------

def _reachable_start(rng, ctrl, z=0.01625):
    lim = ctrl._limits
    while True:
        q = rng.uniform(lim[:, 0], lim[:, 1])
        p = ctrl.ee_pose(q)[:3, 3]
        if abs(p[2] - z) < 0.03 and np.hypot(*p[:2]) > 0.12:
            return q


def test_chunked_solve_matches_one_long_call_for_env_sized_steps():
    """Stopping early on a stalled chunk must not change what a 2 cm step solves to.

    The reference is the library's single call (200 iterations). Chunks carry only the joint vector, so a
    solve that converges returns the very same angles; a sweep over 2 cm steps loses no successes.
    """
    rng = np.random.default_rng(0)
    long = EEController(chunk_iters=200, max_chunks=1, min_contraction=1e9)
    fast = EEController()
    lost = same = n = 0
    for _ in range(120):
        q = _reachable_start(rng, fast)
        ang = rng.uniform(0, 2 * np.pi)
        tgt = fast.ee_pose(q)[:2, 3] + 0.02 * np.array([np.cos(ang), np.sin(ang)])
        for c in (long, fast):
            c.reset(q0=q, z_push=0.01625)
        ql, okl = long.solve(tgt)
        qf, okf = fast.solve(tgt)
        n += 1
        lost += okl and not okf
        same += okl == okf and np.allclose(ql, qf, atol=1e-9)
    assert lost == 0
    assert same == n


def test_unreachable_target_fails_after_a_few_iterations():
    """A target far outside the workspace must give up in a handful of iterations, not the library's 200."""
    import control.kinematics.ik as ik
    calls = [0]
    real = ik.dls_operator

    def counted(*a, **k):
        calls[0] += 1
        return real(*a, **k)

    ik.dls_operator = counted
    try:
        c = EEController()
        c.reset(q0=np.zeros(5), z_push=0.01625)
        q, ok = c.solve([0.9, 0.0])
    finally:
        ik.dls_operator = real
    assert not ok and q == pytest.approx(np.zeros(5))
    assert calls[0] <= 12
