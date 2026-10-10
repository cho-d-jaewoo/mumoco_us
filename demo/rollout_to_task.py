"""Turn a saved policy rollout (demo/rollout.py) into a task file for the experiment (scenario D), so the
experiment replays what pi0.5 did without the policy server.

    python3 -m demo.rollout_to_task demo/rollouts/pnp/ft_v1/rollout_3.hdf5      # -> tasks/pnp_policy.json
    python3 sim_task.py pnp_policy                                               # its task video

The measured joint path becomes waypoints: between the gripper commands of the policy it is simplified
(Ramer-Douglas-Peucker) to the fewest points that stay within `--tolerance` rad of it on every joint. That
keeps the shape of the motion, drops the pauses while the policy was queried and its small jitter, and keeps
long straight stretches as single segments (many close waypoints would make the replay slow). main.py
drives the waypoints like every other task: smooth through them, stopping only for the gripper.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from mumoco.config import EXPERIMENT_TASKS, JOINT_TOL, TASK_DIR
from mumoco.utils import ask_yes_no, load_task, save_task

TOLERANCE = 0.01                # [rad] largest deviation of the waypoint path from the rollout, per joint


def simplify(q, tolerance):
    """Indices of the points kept by Ramer-Douglas-Peucker on the joint path q (N,7), first and last included."""
    keep, stack = {0, len(q) - 1}, [(0, len(q) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        line = q[b] - q[a]                                       # closest point on the straight line a -> b
        s = np.clip((q[a + 1:b] - q[a]) @ line / max(line @ line, 1e-12), 0.0, 1.0)[:, None]
        error = np.max(np.abs(q[a + 1:b] - (q[a] + s * line)), axis=1)
        if error.max() > tolerance:
            k = a + 1 + int(np.argmax(error))
            keep.add(k)
            stack += [(a, k), (k, b)]
    return sorted(keep)


def rollout_waypoints(q, closed, tolerance=TOLERANCE):
    """Waypoints from the measured joint path q (T,7) and the commanded gripper (T, True = closed): each
    stretch with one gripper command is simplified; its last point carries the gripper state that follows."""
    changes = [0, *(np.flatnonzero(closed[1:] != closed[:-1]) + 1), len(q) - 1]
    waypoints = [{"joint_positions": q[0], "gripper_open": not closed[0]}]
    for a, b in zip(changes[:-1], changes[1:]):
        for i in simplify(q[a:b + 1], tolerance)[1:]:
            last = waypoints[-1]
            if last["gripper_open"] == (not closed[a + i]) and np.max(np.abs(q[a + i] - last["joint_positions"])) < JOINT_TOL:
                continue                                     # the robot stood still (e.g. after the last command)
            waypoints.append({"joint_positions": q[a + i], "gripper_open": not closed[a + i]})
    return waypoints


def main():
    parser = argparse.ArgumentParser(description="Make a task file (experiment scenario D) from a saved policy rollout.")
    parser.add_argument("rollout", type=Path, help="demo/rollouts/<task>/<tag>/rollout_<n>.hdf5")
    parser.add_argument("--name", help="task name (default: <task>_<scenario D name in config.py>, e.g. pnp_policy)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE,
                        help=f"[rad] largest deviation of the waypoints from the rollout path (default {TOLERANCE})")
    args = parser.parse_args()
    with h5py.File(args.rollout, "r") as f:
        q = f["obs/joint_state"][:]
        closed = f["actions/gripper_action"][:] > 0.5      # 1 = closed (DROID)
        info = {key: f.attrs[key] for key in ("tag", "success", "task_description") if key in f.attrs}
    task = args.rollout.parent.parent.name                  # demo/rollouts/<task>/<tag>/...
    short = {s: errors.get("D") for s, errors in EXPERIMENT_TASKS.values()}
    name = args.name or f"{task}_{short.get(task) or 'policy'}"
    waypoints = rollout_waypoints(q, closed, args.tolerance)
    print(f"[INFO] {args.rollout}: {len(q)} samples -> {len(waypoints)} waypoints, "
          f"{int(np.sum(closed[1:] != closed[:-1]))} gripper changes, {info}")
    if (TASK_DIR / f"{name}.json").exists() and not ask_yes_no(f"tasks/{name}.json exists. Overwrite?"):
        return
    save_task(name, waypoints)
    load_task(TASK_DIR / f"{name}.json")                    # raises if a waypoint is outside the joint limits
    print(f"[INFO] Task video: python3 sim_task.py {name}")


if __name__ == "__main__":
    main()
