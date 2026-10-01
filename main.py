"""Experiment GUI: run a saved task and record physical corrections.

Flow: connect -> home -> select task -> select modality -> home -> execute
      -> START (or Ctrl+C) -> physical correction -> START again -> save? -> home -> select task ...
"""

from mumoco.gui import ExperimentUI
from mumoco.utils import Robot, execute_with_physical_correction, load_tasks, save_correction


def display_name(task):
    return task["name"].replace("_", " ").title()


def physical_only(robot, task, ui):
    """Returns the title of the next screen."""
    def request_correction():
        robot.correction_requested = True                # same flag as Ctrl+C -> move_joint -> guide()

    shown = []

    def tick():                                          # called by robot.guide() in correction mode
        if not shown:
            ui.show_correction()
            shown.append(True)
        for command in ui.pop_commands():
            if command == "finish":
                ui.status("Physical Correction", "Finishing correction...")
                return True
            robot.request_gripper(command == "open")
        return False

    ui.show_execution(display_name(task), request_correction)
    samples = execute_with_physical_correction(robot, task["waypoints"], tick)
    if samples is None:
        return "Task Completed"
    if not ui.ask_yes_no("Save Correction?"):
        return "Correction Discarded"
    ui.status("Saving Correction", "Please wait...")
    return "Correction Saved" if save_correction(task["name"], "physical", samples) else "Nothing to Save"


# Implement Language-Only / Multimodal as functions(robot, task, ui) and register them here.
MODALITIES = {"Physical-Only": physical_only, "Language-Only": None, "Multimodal": None}


def experiment(ui):
    ui.status("Robot Initialization", "Connecting to robot...\n\nArm: waiting\nGripper: waiting")
    robot = Robot(should_stop=lambda: ui.closing)
    try:
        ui.status("Robot Initialization", "Arm: Connected\nGripper: Connected\n\nReturning to home...", "moving")
        robot.go_home()
        names = list(MODALITIES)
        unavailable = [i for i, name in enumerate(names) if MODALITIES[name] is None]
        while True:
            tasks = load_tasks()
            if not tasks:
                ui.error("No valid tasks in tasks/. Record one with record_tasks.py.")
                return
            task = tasks[ui.choose("Select Task", [display_name(t) for t in tasks])]
            modality = ui.choose("Correction Modality", names, disabled=unavailable, back=True,
                                 scenario=task["name"], subtitle=task["name"])
            if modality is None:
                continue
            ui.status("Preparing Task", "Returning to home position...", "moving")
            robot.go_home()
            result = MODALITIES[names[modality]](robot, task, ui)
            ui.status(result, "Returning to home position...", "moving")
            robot.go_home()
    finally:
        robot.close()


if __name__ == "__main__":
    ExperimentUI().run(experiment)
