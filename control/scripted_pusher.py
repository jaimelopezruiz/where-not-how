"""C5.1: scripted pusher, a closed-loop tracking controller on the object trajectory.

Pure numpy, no simulator. Interface matches PushTrack-v0:
  obs    = [ee_x, ee_y, cube_x, cube_y, sin_yaw, cos_yaw, ...]   (robot frame, metres)
  action = EE displacement for one control step (dx, dy), metres, norm-clipped to max_delta.

Each step: advance the progress along the reference (furthest projection of the cube, never back),
take a lookahead point a fixed arc length ahead, and push the cube along d = unit(lookahead - cube).
The pusher goal sits behind the box on the ray from the cube centre along -d, so the push line passes
through the centre. If the pusher is not behind the box it goes around on a circle clear of the box
corners (tangent line onto the circle, then along it), never through the footprint.
"""
import numpy as np

from eval.metrics import _polyline_project

APPROACH, PUSH, DONE = "approach", "push", "done"


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def exit_distance(d, yaw, half_side):
    """Distance from the box centre along -d to where the ray leaves the box footprint."""
    u = _rot(yaw).T @ (-np.asarray(d, float))
    return half_side / np.max(np.abs(u))


def contact_point(cube_xy, yaw, d, half_side):
    """Point on the box boundary hit by the ray from the cube centre along -d."""
    return np.asarray(cube_xy, float) - exit_distance(d, yaw, half_side) * np.asarray(d, float)


def box_distance(p, cube_xy, yaw, half_side):
    """Distance from p to the box (0 inside)."""
    q = _rot(yaw).T @ (np.asarray(p, float) - np.asarray(cube_xy, float))
    out = np.maximum(np.abs(q) - half_side, 0.0)
    return float(np.hypot(*out))


class ScriptedPusher:
    """Lookahead pusher.

    Parameters
    ----------
    max_delta : per-step EE displacement limit, m (norm, so each component is also within it)
    lookahead : arc length of the lookahead point ahead of the cube's progress, m
    cube_side : box side, m; None reads the measured value from data/props.yaml
    pusher_radius : capsule radius of the pusher, m (the scene value)
    margin : gap kept between pusher surface and box on the approach goal and on the circle, m
    push_speed : speed while closing the last stretch and while pushing, m/step
    stop_tol : the cube is done when within this of the path end, m
    lat_tol, lat_exit : lateral offset from the push line to start / abandon a push, m
    progress_window : how far ahead of the current progress the cube is searched on the path, m
    stuck_cmd : commands longer than this that move the EE by under a quarter of their length halve the step size, m
    """

    def __init__(self, max_delta=0.02, lookahead=0.025, cube_side=None, pusher_radius=0.006,
                 margin=0.005, push_speed=0.0033, stop_tol=0.005, lat_tol=0.012, lat_exit=0.025,
                 progress_window=0.10, k_lat=0.5, stuck_cmd=0.005):
        if cube_side is None:
            from capture.common import load_props
            cube_side = load_props(require=("cube_side",))["cube_side"]
        self.max_delta, self.lookahead = max_delta, lookahead
        self.half_side = cube_side / 2
        self.half_diag = self.half_side * np.sqrt(2)
        self.radius, self.margin = pusher_radius, margin
        self.push_speed, self.stop_tol = push_speed, stop_tol
        self.lat_tol, self.lat_exit = lat_tol, lat_exit
        self.window, self.k_lat, self.stuck_cmd = progress_window, k_lat, stuck_cmd
        self.clear = self.half_diag + pusher_radius + margin   # circle radius around the cube centre
        self.ref = None
        self.reset([[0.0, 0.0], [1.0, 0.0]])

    # -- reference path -------------------------------------------------------------------------
    def reset(self, ref_xy):
        ref = np.asarray(ref_xy, float).reshape(-1, 2)
        keep = np.r_[True, np.hypot(*np.diff(ref, axis=0).T) > 1e-9]
        ref = ref[keep]
        if len(ref) < 2:
            raise ValueError("reference path needs at least 2 distinct points")
        self.ref = ref
        self._cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ref, axis=0).T))]
        self.total = float(self._cum[-1])
        self.progress = 0.0          # arc length, m
        self.mode = APPROACH
        self.goal = None
        self.d = None
        self.lookahead_pt = ref[0].copy()
        self._orbit = 1.0            # side of the circle taken when the way round is ambiguous
        self._last_ee = None         # previous EE position and command, to notice a step the arm did not take
        self._last_cmd = np.zeros(2)
        self._scale = 1.0            # step size factor, reduced while the arm fails to follow (IK rejects long steps)

    def point_at(self, s):
        s = float(np.clip(s, 0.0, self.total))
        return np.array([np.interp(s, self._cum, self.ref[:, 0]), np.interp(s, self._cum, self.ref[:, 1])])

    def tangent_at(self, s):
        i = int(np.clip(np.searchsorted(self._cum, s, side="right") - 1, 0, len(self.ref) - 2))
        v = self.ref[i + 1] - self.ref[i]
        return v / np.hypot(*v)

    def _advance(self, cube):
        """Furthest projection of the cube onto the path, searched only from the current progress forward."""
        end = min(self.progress + self.window, self.total)
        if end - self.progress < 1e-9:
            return
        inner = self._cum[(self._cum > self.progress) & (self._cum < end)]
        arcs = np.r_[self.progress, inner, end]
        sub = np.array([self.point_at(a) for a in arcs])
        arc, _, _ = _polyline_project(sub, cube[None, :])
        self.progress = min(self.progress + float(arc[0]), self.total)

    # -- control --------------------------------------------------------------------------------
    def __call__(self, obs):
        return self.act(obs)

    def act(self, obs):
        obs = np.asarray(obs, float)
        ee, cube = obs[0:2], obs[2:4]
        yaw = np.arctan2(obs[4], obs[5])
        self._advance(cube)
        self._track_following(ee)
        end = self.ref[-1]
        if self.total - self.progress <= self.lookahead and np.hypot(*(cube - end)) < self.stop_tol:
            self.mode = DONE
            return np.zeros(2)

        self.lookahead_pt = self.point_at(self.progress + self.lookahead)
        v = self.lookahead_pt - cube
        n = np.hypot(*v)
        d = v / n if n > 1e-6 else self.tangent_at(self.progress)
        self.d = d
        nrm = np.array([-d[1], d[0]])
        t = exit_distance(d, yaw, self.half_side)
        self.goal = cube - (t + self.radius + self.margin) * d

        rel = ee - cube
        a, lat = -(rel @ d), rel @ nrm          # distance behind the cube centre, offset from the push line
        gap = a - (t + self.radius)             # room left before the pusher touches the rear face
        if self.mode == PUSH:
            push = a > 0 and abs(lat) <= self.lat_exit and gap > -0.01
        else:
            push = a > 0 and abs(lat) <= self.lat_tol and gap >= 0
        self.mode = PUSH if push else APPROACH
        vec = self._push(d, nrm, lat, gap) if push else self._approach(rel, d)
        norm = np.hypot(*vec)
        vec = vec if norm <= self.max_delta else vec * (self.max_delta / norm)
        self._last_cmd = vec = vec * self._scale
        return vec

    def _track_following(self, ee):
        """Shorten steps while the arm does not execute them (the IK refuses some long steps near joint limits)."""
        if self._last_ee is not None:
            cmd, moved = np.hypot(*self._last_cmd), np.hypot(*(ee - self._last_ee))
            if cmd > self.stuck_cmd and moved < 0.25 * cmd:
                self._scale = max(0.25, self._scale * 0.5)
            elif moved >= 0.5 * cmd:
                self._scale = min(1.0, self._scale * 1.25)
        self._last_ee = ee.copy()

    def _push(self, d, nrm, lat, gap):
        along = np.clip(gap - 0.02, self.push_speed, self.max_delta)
        return along * d + np.clip(-self.k_lat * lat, -along, along) * nrm

    def _approach(self, rel, d):
        """Step towards the goal behind the box without entering the footprint (cube-centred polar frame)."""
        m, clear = self.max_delta, self.clear
        rho = np.hypot(*rel)
        phi = np.arctan2(rel[1], rel[0])
        delta = _wrap(np.arctan2(-d[1], -d[0]) - phi)
        if abs(delta) > np.pi - 0.3:            # way round is ambiguous: keep the side already chosen
            sigma = self._orbit
        else:
            sigma = self._orbit = 1.0 if delta >= 0 else -1.0
        radial = rel / rho if rho > 1e-9 else -d
        tang = sigma * np.array([-radial[1], radial[0]])
        eps = 1e-3
        if rho < clear - eps:                   # inside the circle: move out, and round it once clear of the box
            out = min(m, clear - rho)
            round_ = np.sqrt(max(m * m - out * out, 0.0)) if rho >= self.half_diag + self.radius else 0.0
            return out * radial + min(round_, rho * abs(delta)) * tang
        if rho <= clear + eps:                  # on the circle: follow it (chord stays within 0.6 mm of the arc)
            ang = min(abs(delta), 2 * np.arcsin(min(m / (2 * clear), 1.0)))
            q = clear * np.array([np.cos(phi + sigma * ang), np.sin(phi + sigma * ang)])
            return q - rel
        # outside: head for the circle point at most one tangent angle round, so the line never cuts the circle
        ang = min(abs(delta), np.arccos(clear / rho))
        q = clear * np.array([np.cos(phi + sigma * ang), np.sin(phi + sigma * ang)])
        return q - rel
