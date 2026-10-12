"""Replay a saved policy rollout (demo/rollout.py) on the robot, without the policy server or the cameras.

    python3 -m demo.replay demo/rollouts/pnp/pnp_pnp_v1_9999_20261012_153000.json
    python3 -m demo.replay tasks/pnp_policy.json

From home, the robot follows the recorded joint trajectory on its own time axis and repeats the gripper
commands where the policy gave them (Robot.replay), exactly as the experiment runs scenario D. Ctrl+C stops it.
"""

import argparse
from pathlib import Path

from mumoco.utils import Robot, load_task


def main():
    parser = argparse.ArgumentParser(description="Replay a saved policy rollout on the robot.")
    parser.add_argument("rollout", type=Path, help="rollout or task file with a trajectory")
    args = parser.parse_args()
    task = load_task(args.rollout)
    if "trajectory" not in task:
        parser.error(f"{args.rollout} has no trajectory (a waypoint task: run it with main.py)")
    print(f"[INFO] {task['name']}: {task['trajectory']['t'][-1]:.1f} s, {len(task['trajectory']['t'])} samples")
    robot = Robot()
    try:
        robot.go_home()
        while input("\nSet up the scene, then Enter = replay, q = quit: ").strip().lower() != "q":
            robot.execute(task)
            print("[INFO] Replay finished.")
            robot.go_home()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
