"""Trajectory processing: active-interval slicing of physical corrections."""

import numpy as np

from config import MOTION_THRESHOLD


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

