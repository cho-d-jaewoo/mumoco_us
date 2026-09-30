"""Trajectory processing: active-interval slicing and task waypoint extraction."""

import numpy as np

from config import MOTION_THRESHOLD


def gripper_events(gripper):
    """Frames k where the gripper state differs from frame k-1."""
    g = np.asarray(gripper, dtype=bool)
    return (np.flatnonzero(g[1:] != g[:-1]) + 1).tolist()


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


def mandatory_indices(gripper):
    """Start, end and every gripper event: these frames always become waypoints."""
    return sorted({0, len(gripper) - 1, *gripper_events(gripper)})


def waypoint_indices(q, gripper, n):
    """`n` chronological frame indices: the mandatory frames plus extra frames spread evenly
    along the joint-space path between them."""
    keep = mandatory_indices(gripper)
    if not len(keep) <= n <= len(q):
        raise ValueError(f"n must be between {len(keep)} and {len(q)}")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))))
    segments = list(zip(keep[:-1], keep[1:]))
    count = [0] * len(segments)
    for _ in range(n - len(keep)):                         # each extra point goes to the widest-spaced segment
        j = max((j for j, (a, b) in enumerate(segments) if count[j] < b - a - 1),
                key=lambda j: (arc[segments[j][1]] - arc[segments[j][0]]) / (count[j] + 1))
        count[j] += 1

    extra = []
    for (a, b), k in zip(segments, count):
        if k == 0:
            continue
        targets = arc[a] + (arc[b] - arc[a]) * np.arange(1, k + 1) / (k + 1)
        idx = np.clip(np.searchsorted(arc, targets), a + 1, b - 1)
        if len(set(idx.tolist())) < k:                     # (almost) no motion here -> spread evenly in time
            idx = np.floor(np.linspace(a, b, k + 2)[1:-1] + 0.5).astype(int)
        extra += idx.tolist()
    return sorted(keep + extra)
