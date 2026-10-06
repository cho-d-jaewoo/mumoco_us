"""Experiment GUI: run a saved task and record physical, language or multimodal corrections.

Flow: connect -> home -> user ID -> select task -> select error scenario (A-D) and modality -> home -> instructions
      -> START: execute -> START: correction -> START: finish -> (speech to text) -> save? -> home -> select task ...
Correction: physical = guide the robot; language = the robot stands still while the microphone records;
multimodal = guide the robot while the microphone records.
Participants see only the high-level tasks and scenario letters (EXPERIMENT_TASKS in config.py); the error
names stay internal. A task/scenario/modality is completed once its correction is saved (this session only);
completed ones, scenarios with all modalities completed and tasks with all scenarios completed cannot be selected.
"""

import time

from mumoco.config import EXPERIMENT_TASKS, MODALITY_NAMES
from mumoco.gui import ExperimentUI
from mumoco.microphone import Microphone, Transcriber
from mumoco.utils import Robot, execute_with_correction, find_task_path, load_task, save_correction


def correct_task(robot, task, ui, user, modality, mic, transcriber, title):
    """Run the task and record one correction (modality: physical / language / multimodal); title is the
    high-level task name shown to the participant. Returns (title of the next screen, whether a correction was saved)."""
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
        samples, stop = [], {"t": time.monotonic(), "gripper_open": robot.gripper_open}
        if mic:
            mic.start()                                  # before guide(): speech while it switches modes counts
        try:
            if modality == "language":
                stop["q"] = robot.hold(tick)
            else:
                robot.guide(lambda t, state, gripper_open: samples.append((t, state, gripper_open)), tick)
        finally:
            recording = mic.stop() if mic else None
        return samples, recording, stop

    ui.show_execution(title, request_correction)
    result = execute_with_correction(robot, task["waypoints"], correct)
    if result is None:
        return "Task Completed", False
    samples, recording, stop = result
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
    if save_correction(task["name"], modality, user, stop, samples, speech, recording) is None:
        return "Nothing to Save", False
    return "Correction Saved", True


MODALITIES = list(MODALITY_NAMES)                     # physical, language, multimodal

# Shown before each run; the participant presses START to begin. {task} is the high-level task name.
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
        completed = set()                                # (short name, error, modality) with a saved correction
        titles = list(EXPERIMENT_TASKS)

        def implemented(title):                          # letter -> error, for scenarios with a task file
            short, errors = EXPERIMENT_TASKS[title]
            return {letter: error for letter, error in errors.items() if error and find_task_path(f"{short}_{error}")}

        def task_done(title):
            short = EXPERIMENT_TASKS[title][0]
            return all((short, error, m) in completed for error in implemented(title).values() for m in MODALITIES)

        while True:
            done = [i for i, title in enumerate(titles) if task_done(title)]
            if len(done) == len(titles):
                ui.status("All Tasks Completed", "Thank you for participating!")
                return
            title = titles[ui.choose("Select Task", titles, completed=done)]
            short, errors = EXPERIMENT_TASKS[title]
            letters, ready = list(errors), implemented(title)
            choice = ui.choose_scenario(
                "Choose Error Scenario and Correction Modality", title, letters,
                [MODALITY_NAMES[m] for m in MODALITIES],
                unavailable=[i for i, letter in enumerate(letters) if letter not in ready],
                completed=[(i, j) for i, letter in enumerate(letters) for j, m in enumerate(MODALITIES)
                           if (short, errors[letter], m) in completed],
                example=f"{short}_answer")
            if choice is None:
                continue
            letter, modality = letters[choice[0]], MODALITIES[choice[1]]
            error = errors[letter]
            task = load_task(find_task_path(f"{short}_{error}"))
            print(f"[INFO] {title}, scenario {letter} = {task['name']}, {MODALITY_NAMES[modality]}")
            ui.status("Preparing Task", "Returning to home position...", "moving")
            robot.go_home()
            ui.wait_for_start("Instructions", INSTRUCTIONS[MODALITY_NAMES[modality]].format(task=title))
            result, saved = correct_task(robot, task, ui, user, modality, mic, transcriber, title)
            if saved:
                completed.add((short, error, modality))
            ui.status(result, "Returning to home position...", "moving")
            robot.go_home()
    finally:
        if mic:
            mic.close()
        robot.close()


if __name__ == "__main__":
    ExperimentUI().run(experiment)
