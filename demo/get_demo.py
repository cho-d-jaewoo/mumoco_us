"""Demonstrations for pi0.5 fine-tuning: replay recorded task waypoints with the cameras on.

    python3 -m demo.cameras_publisher         # (or the lab's codata publisher: same topics)
    python3 -m demo.get_demo pnp              # demo/demos/pnp/*.json -> demo/data/pnp/demo_<n>.hdf5
    python3 -m demo.get_demo pnp --json pnp_spill --repeat 5    # only the given JSON(s), 5 demos each

Each JSON (made with record_tasks.py, moved to demo/demos/<task>/) is replayed exactly as main.py executes a
task: from home with the gripper open, smooth through the waypoints, stopping only for the gripper. Meanwhile a
thread samples at record_freq the newest arm state, the velocity command just sent, the gripper and the newest
camera frames. HDF5 layout as the lab's get_demo.py (codata / GELLO):

    obs/joint_state (T,7) [rad]           obs/ee_state (T,7) TCP xyz [m] + quaternion xyzw
    obs/gripper_state (T)                 obs/timestep (T) [s] since the recording started
    obs/<camera topic> (T,H,W,3) uint8 RGB or (T,H,W) uint16 depth
    actions/joint_vel (T,7) [rad/s] the velocity command sent to the NUC
    actions/ee_vel (T,6) J @ joint_vel    actions/gripper_action (T) the commanded gripper
    attrs: fps, record_freq, task_description, labels, source (the replayed JSON)

Gripper values follow DROID: 1 = closed, 0 = open.
"""

import argparse
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from mumoco.config import ROOT
from mumoco.franka import Franka
from mumoco.utils import Robot, ask_yes_no, load_task
from .cameras_subscriber import CamerasSubscriber

CONFIG = Path(__file__).parent / "config" / "get_demo.yaml"
STALE = 0.5                     # [s] a camera frame older than this is reported


def quaternion(R):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def record_rollout(robot, waypoints, cameras, record_freq, end_hold):
    """Replay the waypoints while sampling at record_freq; returns (observations, actions, stale samples)."""
    observations = {key: [] for key in ("joint_state", "ee_state", "gripper_state", "timestep", *cameras.topics)}
    actions = {"joint_vel": [], "ee_vel": [], "gripper_action": []}
    done, stale = threading.Event(), {"count": 0}

    def sample():
        t0, k = time.monotonic(), 0
        while not done.is_set():
            state, frames = robot.last_state, cameras.get_last_frames()
            if state is not None and len(frames) == len(cameras.topics):
                q, qdot = np.array(state["q"]), np.array(robot.last_qdot)
                position, rotation = Franka.joint2pose(q)
                command = robot.gripper_open if robot.gripper_command is None else robot.gripper_command
                observations["joint_state"].append(q)
                observations["ee_state"].append(np.concatenate((position, quaternion(rotation))))
                observations["gripper_state"].append(0.0 if robot.gripper_open else 1.0)
                observations["timestep"].append(time.monotonic() - t0)
                for topic in cameras.topics:
                    observations[topic].append(frames[topic])
                actions["joint_vel"].append(qdot)
                actions["ee_vel"].append(state["J"] @ qdot)
                actions["gripper_action"].append(0.0 if command else 1.0)
                stale["count"] += max(cameras.ages().values()) > STALE
            k += 1
            time.sleep(max(0.0, t0 + k / record_freq - time.monotonic()))

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        robot.execute_task(waypoints)
        time.sleep(end_hold)
    finally:
        done.set()
        thread.join()
    return observations, actions, stale["count"]


def save_to_hdf5(path, observations, actions, attrs):
    """obs/ and actions/ groups as the lab's codata save_to_hdf5; images compressed (lzf), one chunk per frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for group, data in (("obs", observations), ("actions", actions)):
            g = f.create_group(group)
            for key, values in data.items():
                array = np.asarray(values)
                image = array.ndim >= 3
                g.create_dataset(key, data=array, compression="lzf" if image else None,
                                 chunks=(1, *array.shape[1:]) if image else None)
        for key, value in attrs.items():
            f.attrs[key] = value


def next_index(save_dir):
    indices = [int(path.stem.split("_")[-1]) for path in save_dir.glob("demo_*.hdf5") if path.stem.split("_")[-1].isdigit()]
    return max(indices, default=-1) + 1


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    parser = argparse.ArgumentParser(description="Record demonstrations by replaying task waypoints with the cameras on.")
    parser.add_argument("task", choices=list(cfg["tasks"]), help="task name (demo/demos/<task>/*.json)")
    parser.add_argument("--json", nargs="+", metavar="NAME", help="only these JSONs of the task (default: all)")
    parser.add_argument("--repeat", type=int, default=1, help="replays of each JSON (default: 1)")
    args = parser.parse_args()
    json_dir, save_dir = ROOT / cfg["json_dir"] / args.task, ROOT / cfg["save_dir"] / args.task
    if args.json:
        files = [json_dir / (name if name.endswith(".json") else f"{name}.json") for name in args.json]
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            parser.error(f"not in {json_dir}/: {', '.join(missing)}")
    else:
        files = sorted(json_dir.glob("*.json"))
    if not files:
        parser.error(f"no waypoint files in {json_dir}/ (record them with record_tasks.py)")
    description = cfg["tasks"][args.task]["task_description"]
    print(f"[INFO] {len(files)} waypoint files x {args.repeat}, task description: \"{description}\"")

    sub = cfg["camera_subscriber"]
    cameras = CamerasSubscriber(sub["topics"], sub["server_addr"], sub["port"])
    cameras.start_thread()
    missing = cameras.wait_for_topics()
    if missing:
        cameras.close_subscriber()
        raise SystemExit(f"[ERROR] No frames from {', '.join(missing)}. Is the camera publisher running?")
    robot = Robot()
    try:
        robot.go_home()
        runs = [path for path in files for _ in range(args.repeat)]
        for i, path in enumerate(runs, 1):
            answer = input(f"\n[{i}/{len(runs)}] {path.name}: set up the scene, then Enter = record, "
                           "s = skip, q = quit: ").strip().lower()
            if answer == "q":
                break
            if answer == "s":
                continue
            waypoints = load_task(path)["waypoints"]
            observations, actions, stale = record_rollout(robot, waypoints, cameras, cfg["record_freq"], cfg["end_hold"])
            count = len(observations["joint_state"])
            print(f"[INFO] {count} samples ({count / cfg['record_freq']:.1f} s)")
            if stale:
                print(f"[WARNING] {stale} samples had a camera frame older than {STALE} s.")
            if count and ask_yes_no("Save this demo?"):
                save_path = save_dir / f"demo_{next_index(save_dir)}.hdf5"
                save_to_hdf5(save_path, observations, actions,
                             {"fps": cfg["record_freq"], "record_freq": cfg["record_freq"],
                              "task_description": description, "labels": [], "source": path.name})
                print(f"[INFO] Saved {save_path}")
            robot.go_home()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        robot.close()
        cameras.close_subscriber()


if __name__ == "__main__":
    main()
