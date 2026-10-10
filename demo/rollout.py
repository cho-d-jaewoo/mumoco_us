"""pi0.5 rollouts on the robot: the policy runs on a GPU server (openpi scripts/serve_policy.py, e.g. on ARC),
this script sends the camera images and the robot state and executes the returned joint velocity chunks.

    python3 -m demo.cameras_publisher                       # (other terminal)
    python3 -m demo.rollout pnp --tag base --hz 15          # pi05_droid as is (DROID rate)
    python3 -m demo.rollout pnp --tag ft                    # fine-tuned on our demos (20 Hz, settings: get_demo.yaml)

Each rollout starts at home with the gripper open, like the demos. Ctrl+C ends it early; then it asks whether
it succeeded and whether to save it as <save_dir>/<task>/<tag>/rollout_<n>.hdf5 (the demo HDF5 layout; attrs
also hold success, tag and the policy settings), so base and fine-tuned rollouts can be compared.

Observation as the LeRobot dataset (demo/preprocess_lerobot.py): the same cameras and image size, joint
positions, gripper 1 = closed / 0 = open, and the task's task_description as the prompt. Action: 7 joint
velocities + gripper (> 0.5 = close), as openpi examples/droid/main.py. Needs the openpi client:
    pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"
"""

import argparse
import contextlib
import signal
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from mumoco.config import LOWER, ROOT, UPPER
from mumoco.utils import Robot, ask_yes_no
from .cameras_subscriber import CamerasSubscriber
from .get_demo import STALE, next_index, record_rollout, save_to_hdf5

CONFIG = Path(__file__).parent / "config" / "get_demo.yaml"
LIMIT_MARGIN = 0.05             # [rad] joints this close to a limit are not driven further toward it


def resize(image, size):
    """(H,W,3) uint8 -> resized uint8, exactly as demo/preprocess_lerobot.py prepares the training images."""
    return np.asarray(Image.fromarray(image).resize(tuple(size), Image.BILINEAR))


@contextlib.contextmanager
def delay_ctrl_c():
    """Ctrl+C during a server call would break the connection: handle it after the call (openpi droid/main.py)."""
    interrupted = []
    previous = signal.signal(signal.SIGINT, lambda signum, frame: interrupted.append(True))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        if interrupted:
            raise KeyboardInterrupt


def run_policy(robot, cameras, client, prompt, views, size, rc):
    """Query the policy every open_loop_horizon steps and execute its actions at rc["hz"] until max_time or Ctrl+C."""
    period, chunk, k = 1.0 / rc["hz"], None, 0
    start = next_time = time.monotonic()
    try:
        while time.monotonic() - start < rc["max_time"]:
            q = robot.read_state()["q"]
            robot.update_gripper()
            if chunk is None or k >= min(rc["open_loop_horizon"], len(chunk)):
                robot.send_velocity(np.zeros(7))         # stand still while the server thinks
                frames = cameras.get_last_frames()
                observation = {
                    "observation/exterior_image_1_left": resize(frames[views["exterior_image_1_left"]], size),
                    "observation/wrist_image_left": resize(frames[views["wrist_image_left"]], size),
                    "observation/joint_position": np.asarray(q, dtype=np.float32),
                    "observation/gripper_position": np.array([0.0 if robot.gripper_open else 1.0], dtype=np.float32),
                    "prompt": prompt,
                }
                with delay_ctrl_c():
                    chunk = np.asarray(client.infer(observation)["actions"])
                k = 0
                next_time = time.monotonic()
            action = chunk[k]
            k += 1
            qdot = np.clip(rc["action_scale"] * action[:7], -rc["max_speed"], rc["max_speed"])
            qdot[((q < LOWER + LIMIT_MARGIN) & (qdot < 0)) | ((q > UPPER - LIMIT_MARGIN) & (qdot > 0))] = 0.0
            robot.send_velocity(qdot)
            robot.request_gripper(action[7] <= 0.5)      # 1 = closed (ignored while the gripper still moves)
            next_time += period
            time.sleep(max(0.0, next_time - time.monotonic()))
    except KeyboardInterrupt:
        print("\n[INFO] Rollout stopped.")
    finally:
        robot.send_velocity(np.zeros(7))


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


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    rc = cfg["rollout"]
    parser = argparse.ArgumentParser(description="Run pi0.5 (openpi policy server) on the robot and save the rollouts.")
    parser.add_argument("task", choices=list(cfg["tasks"]), help="task: its task_description is the prompt")
    parser.add_argument("--tag", required=True, help="model name for the saved rollouts, e.g. base or ft_20k")
    parser.add_argument("--prompt", help="instead of the task's task_description")
    for key in ("host", "port", "hz", "open_loop_horizon", "action_scale", "max_speed", "max_time"):
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(rc[key]), default=rc[key],
                            help=f"(default from get_demo.yaml: {rc[key]})")
    args = parser.parse_args()
    rc = {key: getattr(args, key) for key in rc if key != "save_dir"} | {"save_dir": rc["save_dir"]}
    prompt = args.prompt or cfg["tasks"][args.task]["task_description"]
    views, size = cfg["lerobot"], cfg["lerobot"]["image_size"]
    save_dir = ROOT / rc["save_dir"] / args.task / args.tag
    print(f'[INFO] Prompt: "{prompt}", {rc["hz"]} Hz, chunks of {rc["open_loop_horizon"]}, '
          f'scale {rc["action_scale"]}, max {rc["max_speed"]} rad/s')

    from openpi_client import websocket_client_policy
    print(f"[INFO] Connecting to the policy server at {rc['host']}:{rc['port']}...")
    client = websocket_client_policy.WebsocketClientPolicy(rc["host"], rc["port"])
    print(f"[INFO] Server metadata: {client.get_server_metadata()}")

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
            if input("\nSet up the scene, then Enter = run the policy, q = quit: ").strip().lower() == "q":
                break
            observations, actions, stale = record_rollout(
                robot, lambda: run_policy(robot, cameras, client, prompt, views, size, rc), cameras,
                cfg["record_freq"], cfg["end_hold"])
            count = len(observations["joint_state"])
            print(f"[INFO] {count} samples ({count / cfg['record_freq']:.1f} s)")
            if stale:
                print(f"[WARNING] {stale} samples had a camera frame older than {STALE} s.")
            success = ask_success()
            if count and ask_yes_no("Save this rollout?"):
                save_path = save_dir / f"rollout_{next_index(save_dir, 'rollout')}.hdf5"
                save_to_hdf5(save_path, observations, actions,
                             {"fps": cfg["record_freq"], "record_freq": cfg["record_freq"], "task_description": prompt,
                              "labels": [], "source": "policy", "tag": args.tag, "success": success,
                              **{f"policy_{key}": value for key, value in rc.items() if key != "save_dir"}})
                print(f"[INFO] Saved {save_path}")
            robot.go_home()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        robot.close()
        cameras.close_subscriber()


if __name__ == "__main__":
    main()
