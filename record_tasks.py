"""Demonstrate a task by hand-guiding the robot and save the waypoints picked with the joystick in tasks/.

Flow: connect -> home -> Ctrl+C -> guide; X = record waypoint, A/B = open/close gripper
      -> Enter -> preview on the robot -> save?
Keyboard fallback while guiding: type x / o / c + Enter.
"""

from joystick_input import open_joystick
from utils import Robot, ask_task_name, ask_yes_no, read_terminal_line, save_task


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
        pressed = joystick.poll() if joystick else set()
        line = read_terminal_line()
        if "X" in pressed or line == "x":
            record_waypoint()
        if "A" in pressed or line == "o":
            robot.request_gripper(True)
        if "B" in pressed or line == "c":
            robot.request_gripper(False)
        return line == ""                                # Enter alone finishes guiding

    with robot.ctrl_c_requests_correction():
        print("\n[INFO] Press Ctrl+C to enter guiding mode.")
        robot.wait_for_ctrl_c()
        print("[INFO] Activate the External Activation Switch and guide the robot.")
        print("[INFO] X: record waypoint   A: open gripper   B: close gripper")
        print("[INFO] Finish: deactivate the switch, then press Enter. (Ctrl+C: quit)")
        robot.guide(lambda t, state, gripper_open: latest.update(q=state["q"]), tick)
    return waypoints


def main():
    joystick = open_joystick()
    robot = Robot()
    try:
        robot.go_home()
        waypoints = record_waypoints(robot, joystick)
        if not waypoints:
            print("[WARNING] No waypoints were recorded.")
        else:
            print(f"[INFO] Recorded waypoints: {len(waypoints)}")
            input("Press Enter to execute the recorded waypoints.")
            robot.go_home()
            robot.execute_task(waypoints)
            print("[INFO] Preview finished.")
            if ask_yes_no("Save this task?"):
                save_task(ask_task_name(), waypoints)
        robot.go_home()
    finally:
        robot.close()
        if joystick:
            joystick.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
