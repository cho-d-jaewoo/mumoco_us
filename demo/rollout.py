"""pi0.5 rollouts on the robot. The policy runs on a GPU server (openpi scripts/serve_policy.py on ARC, see
demo/arc/serve.sh) reached through an SSH tunnel; this script sends the camera images and the robot state,
executes the joint velocities it returns and records what the robot did.

    python3 -m demo.cameras_publisher                         # (other terminal)
    python3 -m demo.rollout pnp --policy pnp_v1/9999          # the name of the checkpoint the server is serving
    python3 -m demo.rollout pnp --policy base                 # pi05_droid as is

Each rollout starts at home with the gripper open, like the demos. Ctrl+C ends it (or max_time). Then it asks
whether it succeeded and whether to save it as <save_dir>/<task>/<task>_<policy>_<time>.json: the robot
trajectory only (no images), dense enough to replay the rollout later without the policy server
(demo/replay.py; as experiment scenario D after demo/rollout_to_task.py).

Observation as the LeRobot dataset (demo/preprocess_lerobot.py): the same cameras and image size, joint
positions, gripper 1 = closed / 0 = open, and the task's task_description as the prompt. Action: 7 joint
velocities [rad/s] + gripper (> 0.5 = close). Inference runs in the background (demo/policy_client.py).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from mumoco.config import LOWER, ROOT, UPPER
from mumoco.trajectory_utils import rollout_waypoints
from mumoco.utils import Robot, ask_yes_no, safe_name
from .cameras_subscriber import CamerasSubscriber
from .policy_client import PolicyClient

CONFIG = Path(__file__).parent / "config" / "get_demo.yaml"
LIMIT_MARGIN = 0.05             # [rad] joints this close to a limit are not driven further toward it
FIELDS = {
    "t": "[s] since the first policy action; one sample per control tick (settings.hz)",
    "q": "[rad] measured joint positions at the tick",
    "qdot": "[rad/s] joint velocity command sent at the tick (after action_scale, max_speed and the joint limit guard)",
    "gripper_open": "measured gripper state",
    "gripper_command": "gripper state the policy asked for (action[7] <= 0.5 = open); sent when it differs and the gripper is idle",
    "action": "policy action executed at the tick: 7 joint velocities + gripper (1 = closed); null = no action buffered, robot held still",
    "chunk, index": "which action chunk of the rollout the action came from and its position in that chunk",
    "waypoints": "the joint path reduced to waypoints (scenes of the simulation); the replay uses the trajectory",
}


def resize(image, size):
    """(H,W,3) uint8 -> resized uint8, exactly as demo/preprocess_lerobot.py prepares the training images."""
    return np.asarray(Image.fromarray(image).resize(tuple(size), Image.BILINEAR))


def run_policy(robot, cameras, policy, prompt, views, size, rc):
    """Execute the policy at rc["hz"] until max_time or Ctrl+C; returns one sample per control tick and the
    time each action chunk took to arrive [s]."""
    samples, period = [], 1.0 / rc["hz"]
    policy.clear()
    gripper_command = bool(robot.gripper_open)
    start = next_time = time.monotonic()
    try:
        while (t := time.monotonic() - start) < rc["max_time"]:
            if policy.error:
                raise RuntimeError(f"Policy server connection lost: {policy.error}")
            q = robot.read_state()["q"]
            robot.update_gripper()
            frames = cameras.get_last_frames()
            policy.post({
                "observation/exterior_image_1_left": resize(frames[views["exterior_image_1_left"]], size),
                "observation/wrist_image_left": resize(frames[views["wrist_image_left"]], size),
                "observation/joint_position": np.asarray(q, dtype=np.float32),
                "observation/gripper_position": np.array([0.0 if robot.gripper_open else 1.0], dtype=np.float32),
                "prompt": prompt,
            })
            step = policy.next_action()
            qdot = np.zeros(7)                           # no action buffered: stand still
            if step is not None:
                action = step[0]
                qdot = np.clip(rc["action_scale"] * action[:7], -rc["max_speed"], rc["max_speed"])
                qdot[((q < LOWER + LIMIT_MARGIN) & (qdot < 0)) | ((q > UPPER - LIMIT_MARGIN) & (qdot > 0))] = 0.0
                gripper_command = bool(action[7] <= 0.5)
            robot.send_velocity(qdot)
            robot.request_gripper(gripper_command)       # ignored while the gripper still moves
            samples.append({"t": t, "q": q, "qdot": robot.last_qdot, "gripper_open": bool(robot.gripper_open),
                            "gripper_command": gripper_command, "action": None if step is None else step[0],
                            "chunk": None if step is None else step[1], "index": None if step is None else step[2]})
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[INFO] Rollout stopped.")
    finally:
        robot.send_velocity(np.zeros(7))
    latencies = list(policy.latencies)
    policy.clear()                                       # nothing is inferred while the robot stands
    return samples, latencies


def ask_success():
    while True:
        answer = input("Success? (y = 100, n = 0, or 0-100): ").strip().lower()
        try:
            value = {"y": 100.0, "n": 0.0}.get(answer, None)
            value = float(answer) if value is None else value
        except ValueError:
            continue
        if 0 <= value <= 100:
            return value / 100


def save_rollout(path, samples, header):
    """Write the rollout as a task file with a dense trajectory (mumoco.utils.load_task reads it back)."""
    rounded = lambda values, digits: None if values is None else np.round(values, digits).tolist()
    trajectory = [{"t": round(s["t"] - samples[0]["t"], 4), "q": rounded(s["q"], 6), "qdot": rounded(s["qdot"], 5),
                   "gripper_open": s["gripper_open"], "gripper_command": s["gripper_command"],
                   "action": rounded(s["action"], 5), "chunk": s["chunk"], "index": s["index"]} for s in samples]
    waypoints = [{"joint_positions": np.round(w["joint_positions"], 6).tolist(), "gripper_open": w["gripper_open"]}
                 for w in rollout_waypoints([s["q"] for s in samples], [s["gripper_command"] for s in samples])]
    rows = lambda items: ",\n    ".join(json.dumps(item) for item in items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{json.dumps(header, indent=2, default=str)[:-2]},\n  "waypoints": [\n    {rows(waypoints)}\n  ],\n'
                    f'  "trajectory": [\n    {rows(trajectory)}\n  ]\n}}\n')
    return len(trajectory), len(waypoints)


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    rc = cfg["rollout"]
    parser = argparse.ArgumentParser(description="Run pi0.5 (openpi policy server) on the robot and save the rollouts.")
    parser.add_argument("task", choices=list(cfg["tasks"]), help="task: its task_description is the prompt")
    parser.add_argument("--policy", required=True,
                        help="name of the checkpoint the server is serving, e.g. pnp_v1/9999 or base (saved with the rollout)")
    parser.add_argument("--prompt", help="instead of the task's task_description")
    for key in ("host", "port", "hz", "action_scale", "max_speed", "max_time"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(rc[key]), default=rc[key],
                            help=f"(default from get_demo.yaml: {rc[key]})")
    args = parser.parse_args()
    rc = {key: getattr(args, key) for key in ("host", "port", "hz", "action_scale", "max_speed", "max_time")}
    prompt = args.prompt or cfg["tasks"][args.task]["task_description"]
    views, size = cfg["lerobot"], cfg["lerobot"]["image_size"]
    save_dir = ROOT / cfg["rollout"]["save_dir"] / args.task
    print(f'[INFO] Policy {args.policy}, prompt: "{prompt}", {rc["hz"]} Hz, scale {rc["action_scale"]}, '
          f'max {rc["max_speed"]} rad/s')

    policy = PolicyClient(rc["host"], rc["port"])
    print(f"[INFO] Policy server connected at {policy.uri}.")
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
        while True:
            if input("\nSet up the scene, then Enter = run the policy (Ctrl+C ends it), q = quit: ").strip().lower() == "q":
                break
            samples, latencies = run_policy(robot, cameras, policy, prompt, views, size, rc)
            first = next((i for i, s in enumerate(samples) if s["action"] is not None), None)
            if first is None:
                print("[WARNING] The policy returned no action. Nothing to save.")
                robot.go_home()
                continue
            samples = samples[max(first - 1, 0):]              # drop the wait for the first action
            duration = samples[-1]["t"] - samples[0]["t"]
            idle = sum(s["action"] is None for s in samples[1:])
            print(f"[INFO] {duration:.1f} s, {len(latencies)} action chunks, inference {np.mean(latencies):.2f} s "
                  f"(max {np.max(latencies):.2f} s), {idle} of {len(samples)} ticks without an action")
            success = ask_success()
            if ask_yes_no("Save this rollout?"):
                now = datetime.now()
                name = safe_name(f"{args.task}_{args.policy}_{now:%Y%m%d_%H%M%S}")
                header = {"name": name, "task": args.task, "recorded_at": now.isoformat(timespec="seconds"),
                          "success": success, "duration": round(duration, 2),
                          "policy": {"name": args.policy, "prompt": prompt, "server_metadata": policy.metadata},
                          "settings": {key: rc[key] for key in ("hz", "action_scale", "max_speed", "max_time")},
                          "inference": {"chunks": len(latencies), "mean_latency": round(float(np.mean(latencies)), 3),
                                        "max_latency": round(float(np.max(latencies)), 3), "ticks_without_action": idle},
                          "fields": FIELDS}
                count, waypoints = save_rollout(save_dir / f"{name}.json", samples, header)
                print(f"[INFO] Saved {save_dir / f'{name}.json'} ({count} samples, {waypoints} waypoints)")
            robot.go_home()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        robot.close()
        cameras.close_subscriber()
        policy.close()


if __name__ == "__main__":
    main()
