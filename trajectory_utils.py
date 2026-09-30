"""Trajectory processing: active-interval slicing of physical corrections and smooth task trajectories."""

import numpy as np

from config import JOINT_TOL, MOTION_THRESHOLD


def active_interval(q, gripper, threshold=MOTION_THRESHOLD):
    """Inclusive (start, end) of the meaningful interaction, or None if nothing happened.

    A frame is active if any joint moved more than `threshold` since the previous frame,
    or if the gripper state changed. `start` is the frame just BEFORE the first active frame
    (the configuration before motion), `end` is the last active frame.
    """
    q, g = np.asarray(q, dtype=np.float64), np.asarray(gripper, dtype=bool)
    if len(q) < 2:
        return None
    changed = (np.max(np.abs(np.diff(q, axis=0)), axis=1) > threshold) | (g[1:] != g[:-1])
    frames = np.flatnonzero(changed) + 1
    if len(frames) == 0:
        return None
    return int(frames[0] - 1), int(frames[-1])



class JointSpline:
    """Smooth joint trajectory through `points`, at rest at both ends.

    Cubic Hermite segments with Catmull-Rom velocities at the via points: passes exactly through
    every point with continuous velocity, and is slowed down uniformly so that |qdot| <= vmax and
    |qddot| <= amax on every joint. sample(t) -> (q, qdot).
    """

    def __init__(self, points, vmax, amax):
        p = [np.asarray(points[0], dtype=np.float64)]
        for q in points[1:]:                               # points within JOINT_TOL count as the same place
            if np.max(np.abs(np.asarray(q) - p[-1])) > JOINT_TOL:
                p.append(np.asarray(q, dtype=np.float64))
        self.p = np.array(p if len(p) > 1 else p * 2)
        d = np.max(np.abs(np.diff(self.p, axis=0)), axis=1)
        dt = np.maximum(np.maximum(d / vmax, 2 * np.sqrt(d / amax)), 1e-3)  # nominal segment times (speed, acceleration)
        self.knots = np.concatenate(([0.0], np.cumsum(dt)))
        self.v = np.zeros_like(self.p)
        self.v[1:-1] = (self.p[2:] - self.p[:-2]) / (self.knots[2:] - self.knots[:-2])[:, None]

        u = np.linspace(0.0, 1.0, 21)[:, None]
        vpeak = apeak = 0.0
        for k in range(len(self.p) - 1):
            dp, m0, m1, h = self.p[k] - self.p[k + 1], self.v[k] * dt[k], self.v[k + 1] * dt[k], dt[k]
            qd = (6 * (u**2 - u) * dp + (3 * u**2 - 4 * u + 1) * m0 + (3 * u**2 - 2 * u) * m1) / h
            qdd_ends = np.array([-6 * dp - 4 * m0 - 2 * m1, 6 * dp + 2 * m0 + 4 * m1]) / h**2  # qdd is linear
            vpeak, apeak = max(vpeak, np.max(np.abs(qd))), max(apeak, np.max(np.abs(qdd_ends)))
        self.scale = max(1.0, vpeak / vmax, np.sqrt(apeak / amax))
        self.duration = self.knots[-1] * self.scale

    def sample(self, t):
        s = min(max(t / self.scale, 0.0), self.knots[-1])
        k = min(np.searchsorted(self.knots, s, side="right") - 1, len(self.p) - 2)
        h = self.knots[k + 1] - self.knots[k]
        u = (s - self.knots[k]) / h
        p0, p1, m0, m1 = self.p[k], self.p[k + 1], self.v[k] * h, self.v[k + 1] * h
        q = (2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * m0 + (3 * u**2 - 2 * u**3) * p1 + (u**3 - u**2) * m1
        qd = (6 * (u**2 - u) * (p0 - p1) + (3 * u**2 - 4 * u + 1) * m0 + (3 * u**2 - 2 * u) * m1) / h
        return q, qd / self.scale
