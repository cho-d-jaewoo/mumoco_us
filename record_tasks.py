"""Demonstrate a task by hand-guiding the robot and save it as waypoints in tasks/."""

import numpy as np

from trajectory_utils import active_interval, gripper_events, mandatory_indices, waypoint_indices
from utils import Robot, ask_task_name, ask_yes_no, save_task


def record_demonstration(robot):
    """Joint positions and measured gripper state of every arm state message while guiding (nothing else is kept)."""
    q, gripper = [], []
    with robot.ctrl_c_requests_correction():
        print("\n[INFO] Press Ctrl+C to enter guiding mode.")
        robot.wait_for_ctrl_c()
        robot.guide(lambda t, state, gripper_open: (q.append(state["q"]), gripper.append(gripper_open)))
    return np.array(q), np.array(gripper, dtype=bool)


def ask_waypoint_count(q, gripper):
    minimum = len(mandatory_indices(gripper))
    while True:
        answer = input("Number of waypoints: ").strip()
        if not answer.isdigit():
            print("[WARNING] Enter a positive integer.")
        elif int(answer) < minimum:
            print(f"The demonstration requires at least {minimum} waypoints\n"
                  f"because of its start/end points and gripper events.\n"
                  f"Please enter a value >= {minimum}.")
        elif int(answer) > len(q):
            print(f"[WARNING] Only {len(q)} samples are available.")
        else:
            return int(answer)


def main():
    robot = Robot()
    try:
        robot.go_home()
        q, gripper = record_demonstration(robot)
        interval = active_interval(q, gripper)
        if interval is None:
            print("[WARNING] No motion or gripper action was detected.")
        elif ask_yes_no("Extract waypoints from the recorded demonstration?"):
            start, end = interval
            q_active, g_active = q[start:end + 1], gripper[start:end + 1]
            idx = waypoint_indices(q_active, g_active, ask_waypoint_count(q_active, g_active))
            waypoints = [{"joint_positions": q_active[i], "gripper_open": bool(g_active[i])} for i in idx]

            print("[INFO] Demonstration processing completed.")
            print(f"[INFO] Recorded samples: {len(q)}")
            print(f"[INFO] Active samples after slicing: {len(q_active)}")
            print(f"[INFO] Detected gripper events: {len(gripper_events(g_active))}")
            print(f"[INFO] Generated waypoints: {len(waypoints)}")
            input("Trajectory generated successfully. Press Enter to execute it.")

            robot.go_home()
            robot.execute_task(waypoints)
            print("[INFO] Preview finished.")
            if ask_yes_no("Save this task?"):
                save_task(ask_task_name(), waypoints)
        robot.go_home()
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
