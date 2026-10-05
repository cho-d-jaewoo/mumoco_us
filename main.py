"""Experiment GUI: run a saved task and record physical, language or multimodal corrections.

Flow: connect -> home -> user ID -> select task -> select modality -> home -> instructions
      -> START: execute -> START: correction -> START: finish -> (speech to text) -> save? -> home -> select task ...
Correction: physical = guide the robot; language = the robot stands still while the microphone records;
multimodal = guide the robot while the microphone records.
A task/modality pair is completed once its correction is saved (this session only); completed pairs,
and tasks with all modalities completed, can no longer be selected.
"""

from mumoco.gui import ExperimentUI
from mumoco.microphone import Microphone, Transcriber
from mumoco.utils import Robot, execute_with_correction, load_tasks, save_correction


def display_name(task):
    return task["name"].replace("_", " ").title()


def correct_task(robot, task, ui, user, modality, mic, transcriber):
    """Run the task and record one correction (modality: physical / language / multimodal).
    Returns (title of the next screen, whether a correction was saved)."""
    if modality == "physical":
        mic = None

    def request_correction():
        robot.correction_requested = True                # same flag as Ctrl+C -> move_joint -> guide()

    shown = []

    def tick():                                          # called by robot.guide() / robot.hold() in correction mode
        if not shown:
            ui.show_correction(modality)
            shown.append(True)
        for command in ui.pop_commands():
            if command == "finish":
                ui.status("Correction", "Finishing correction...")
                return True
            robot.request_gripper(command == "open")     # language: the screen sends no gripper commands
        return False

    def correct():                                       # the robot has just stopped
        samples = []
        if mic:
            mic.start()                                  # before guide(): speech while it switches modes counts
        try:
            if modality == "language":
                robot.hold(tick)
            else:
                robot.guide(lambda t, state, gripper_open: samples.append((t, state, gripper_open)), tick)
        finally:
            recording = mic.stop() if mic else None
        return samples, recording

    ui.show_execution(display_name(task), request_correction)
    result = execute_with_correction(robot, task["waypoints"], correct)
    if result is None:
        return "Task Completed", False
    samples, recording = result
    speech, text = None, ""
    if mic:
        if recording is None:
            text = "No audio was recorded."
        else:
            ui.status("Processing Speech", "Converting your speech to text...")
            speech = transcriber.transcribe(recording)
            print(f'[INFO] Speech: "{speech["text"]}"')
            text = f'You said:\n"{speech["text"]}"' if speech["text"] else "No speech was recognized."
    if not ui.ask_yes_no("Save Correction?", text):
        return "Correction Discarded", False
    ui.status("Saving Correction", "Please wait...")
    if save_correction(task["name"], modality, user, samples, speech, recording) is None:
        return "Nothing to Save", False
    return "Correction Saved", True


MODALITIES = {"Physical-Only": "physical", "Language-Only": "language", "Multimodal": "multimodal"}

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
                   "press START, physically correct the robot what it should do,\n"
                   "while at the same time telling it what you want through the microphone.\n"
                   "Press START again when you are done.\n\n"
                   "Press START to begin."),
}


def experiment(ui):
    ui.status("Robot Initialization", "Connecting to robot...\n\nArm: waiting\nGripper: waiting")
    robot = Robot(should_stop=lambda: ui.closing)
    mic = None
    try:
        ui.status("Robot Initialization", "Arm: Connected\nGripper: Connected\nMicrophone: waiting")
        mic = Microphone()                               # raises if the microphone is not connected
        mic.start()                                      # raises if it cannot be opened (e.g. held by another program)
        mic.stop()
        ui.status("Robot Initialization", "Arm: Connected\nGripper: Connected\nMicrophone: Connected\n\n"
                  "Loading speech recognition...")
        transcriber = Transcriber()
        ui.status("Robot Initialization", "Arm: Connected\nGripper: Connected\nMicrophone: Connected\n\n"
                  "Returning to home...", "moving")
        robot.go_home()
        user = ui.ask_user_id()
        print(f"[INFO] User ID: {user}")
        completed = set()                                # (task name, modality) with a saved correction
        names = list(MODALITIES)
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
            modality = ui.choose("Correction Modality", names, back=True,
                                 scenario=task["name"], subtitle=task["name"],
                                 completed=[i for i, name in enumerate(names) if (task["name"], name) in completed])
            if modality is None:
                continue
            ui.status("Preparing Task", "Returning to home position...", "moving")
            robot.go_home()
            ui.wait_for_start("Instructions", INSTRUCTIONS[names[modality]].format(task=display_name(task)))
            result, saved = correct_task(robot, task, ui, user, MODALITIES[names[modality]], mic, transcriber)
            if saved:
                completed.add((task["name"], names[modality]))
            ui.status(result, "Returning to home position...", "moving")
            robot.go_home()
    finally:
        if mic:
            mic.close()
        robot.close()


if __name__ == "__main__":
    ExperimentUI().run(experiment)
