"""Experiment GUI: run a saved task and record physical corrections.

Flow: connect -> home -> participant name -> select task -> select modality -> home -> instructions
      -> START: execute -> START: physical correction -> START: finish -> save? -> home -> select task ...
A task/modality pair is completed once its correction is saved (this session only); completed pairs,
and tasks with all modalities completed, can no longer be selected.
"""

from mumoco.gui import ExperimentUI
from mumoco.utils import Robot, execute_with_physical_correction, load_tasks, save_correction


def display_name(task):
    return task["name"].replace("_", " ").title()


def physical_only(robot, task, ui, user):
    """Returns (title of the next screen, whether a correction was saved)."""
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
        return "Task Completed", False
    if not ui.ask_yes_no("Save Correction?"):
        return "Correction Discarded", False
    ui.status("Saving Correction", "Please wait...")
    if save_correction(task["name"], "physical", samples, user) is None:
        return "Nothing to Save", False
    return "Correction Saved", True


# Implement Language-Only / Multimodal as functions(robot, task, ui, user) -> (next title, saved)
# and register them here.
MODALITIES = {"Physical-Only": physical_only, "Language-Only": None, "Multimodal": None}

# Shown before each run; the participant presses START to begin. {task} is the task's display name.
INSTRUCTIONS = {
    "Physical-Only": ("The robot will now perform the task: {task}.\n\n"
                      "Whenever you want to take over or correct the robot,\n"
                      "press START and physically correct the robot,\n"
                      "just enough to show it what it should do.\n"
                      "Press START again when you are done.\n\n"
                      "Press START to begin."),
    "Language-Only": ("The robot will now perform the task: {task}.\n\n"
                      "Whenever you want to correct the robot,\n"
                      "press START and tell the robot what it should do\n"
                      "by speaking into the microphone.\n"
                      "Press START again when you are done.\n\n"
                      "Press START to begin."),
    "Multimodal": ("The robot will now perform the task: {task}.\n\n"
                   "Whenever you want to take over or correct the robot,\n"
                   "press START, physically correct the robot just enough to show it\n"
                   "what it should do, and tell it what you want through the microphone.\n"
                   "Press START again when you are done.\n\n"
                   "Press START to begin."),
}


def experiment(ui):
    ui.status("Robot Initialization", "Connecting to robot...\n\nArm: waiting\nGripper: waiting")
    robot = Robot(should_stop=lambda: ui.closing)
    try:
        ui.status("Robot Initialization", "Arm: Connected\nGripper: Connected\n\nReturning to home...", "moving")
        robot.go_home()
        user = ui.ask_name()
        print(f"[INFO] Participant: {user}")
        completed = set()                                # (task name, modality) with a saved correction
        names = list(MODALITIES)
        unavailable = [i for i, name in enumerate(names) if MODALITIES[name] is None]
        while True:
            tasks = load_tasks()
            if not tasks:
                ui.error("No valid tasks in tasks/. Record one with record_tasks.py.")
                return
            done = [i for i, t in enumerate(tasks) if all((t["name"], name) in completed for name in names)]
            if len(done) == len(tasks):
                ui.status("All Tasks Completed", "Thank you for participating!")
                return
            task = tasks[ui.choose("Select Task", [display_name(t) for t in tasks], completed=done)]
            modality = ui.choose("Correction Modality", names, disabled=unavailable, back=True,
                                 scenario=task["name"], subtitle=task["name"],
                                 completed=[i for i, name in enumerate(names) if (task["name"], name) in completed])
            if modality is None:
                continue
            ui.status("Preparing Task", "Returning to home position...", "moving")
            robot.go_home()
            ui.wait_for_start("Instructions", INSTRUCTIONS[names[modality]].format(task=display_name(task)))
            result, saved = MODALITIES[names[modality]](robot, task, ui, user)
            if saved:
                completed.add((task["name"], names[modality]))
            ui.status(result, "Returning to home position...", "moving")
            robot.go_home()
    finally:
        robot.close()


if __name__ == "__main__":
    ExperimentUI().run(experiment)
