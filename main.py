"""Run a saved task and record physical corrections."""

from utils import Robot, ask_yes_no, choose, execute_with_physical_correction, load_tasks, save_correction


def physical_only(robot, task):
    print(f"\n[INFO] Executing task: {task['name']}")
    print("[INFO] Press Ctrl+C to enter correction mode.")
    samples = execute_with_physical_correction(robot, task["waypoints"])
    if samples is None:
        print("[INFO] Task completed without correction.")
    elif ask_yes_no("Save this correction?"):
        save_correction(task["name"], "physical", samples)
    else:
        print("[INFO] Correction discarded.")


# Implement Language-Only / Multimodal as functions(robot, task) and register them here.
MODALITIES = {"Physical-Only": physical_only, "Language-Only": None, "Multimodal": None}


def select_modality():
    names = list(MODALITIES)
    while (i := choose("Correction modality:", names)) is not None:
        if MODALITIES[names[i]] is not None:
            return MODALITIES[names[i]]
        print(f"[INFO] {names[i]} correction is not implemented yet.")
    return None


def main():
    robot = Robot()
    try:
        robot.go_home()
        while True:
            tasks = load_tasks()
            if not tasks:
                print("[ERROR] No valid tasks in tasks/. Record one with record_tasks.py.")
                break
            i = choose("Available tasks:", [t["name"] for t in tasks])
            if i is None:
                break
            run = select_modality()
            if run is None:
                continue
            robot.go_home()
            run(robot, tasks[i])
            robot.go_home()
    finally:
        robot.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
