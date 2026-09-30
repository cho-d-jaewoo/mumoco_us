"""Demonstrate a task by hand-guiding the robot and save the waypoints picked with the joystick in tasks/.

Flow: connect -> home -> START -> guide; X = record waypoint, A/B = open/close gripper
      -> START -> home -> replay? (Y/X) -> save? (Y/X) -> task name (keyboard)
Keyboard fallback: Ctrl+C / Enter instead of START, x / o / c + Enter while guiding, y / n + Enter for questions.
"""

import time

from joystick_input import open_joystick
from utils import Robot, ask_task_name, read_terminal_line, save_task


def pressed(joystick):
    return joystick.poll() if joystick else set()


def drop_typed_lines():
    while read_terminal_line() is not None:              # keys typed earlier must not answer the next prompt
        pass


def ask(joystick, question):
    """Joystick Y / typed y = yes; joystick X / any other typed line = no."""
    drop_typed_lines()
    print(f"{question} [Y = yes, X = no]: ", end="", flush=True)
    while True:
        buttons, line = pressed(joystick), read_terminal_line()
        if "Y" in buttons or "X" in buttons or line is not None:
            answer = "Y" in buttons or line in ("y", "yes")
            if line is None:
                print("yes" if answer else "no")
            return answer
        time.sleep(0.02)


def record_waypoints(robot, joystick):
    """Each X press stores the current joint positions and the measured gripper state as one waypoint."""
    waypoints, latest = [], {}

    def record_waypoint():
        if robot.gripper_busy():
            print("[WARNING] Gripper is still moving. Waypoint not recorded.")
        elif "q" in latest:
            waypoints.append({"joint_positions": latest["q"].copy(), "gripper_open": robot.gripper_open})
            print(f"[INFO] Waypoint {len(waypoints)} recorded "
                  f"(gripper {'open' if robot.gripper_open else 'closed'}).")

    def tick():
        buttons, line = pressed(joystick), read_terminal_line()
        if "X" in buttons or line == "x":
            record_waypoint()
        if "A" in buttons or line == "o":
            robot.request_gripper(True)
        if "B" in buttons or line == "c":
            robot.request_gripper(False)
        return "START" in buttons or line == ""          # START (or Enter) finishes guiding

    with robot.ctrl_c_requests_correction():
        print("\n[INFO] Press START (or Ctrl+C) to enter guiding mode.")
        while not robot.correction_requested and "START" not in pressed(joystick):
            time.sleep(0.02)
        drop_typed_lines()
        print("[INFO] Activate the External Activation Switch and guide the robot.")
        print("[INFO] X: record waypoint   A: open gripper   B: close gripper")
        print("[INFO] Finish: deactivate the switch, then press START (or Enter).")
        robot.guide(lambda t, state, gripper_open: latest.update(q=state["q"]), tick)
    return waypoints


def main():
    joystick = open_joystick()
    robot = Robot()
    try:
        robot.go_home()
        waypoints = record_waypoints(robot, joystick)
        robot.go_home()
        if not waypoints:
            print("[WARNING] No waypoints were recorded.")
        else:
            print(f"[INFO] Recorded waypoints: {len(waypoints)}")
            if ask(joystick, "Replay the recorded waypoints?"):
                robot.execute_task(waypoints)
                print("[INFO] Replay finished.")
                robot.go_home()
            if ask(joystick, "Save this task?"):
                save_task(ask_task_name(), waypoints)
    finally:
        robot.close()
        if joystick:
            joystick.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
