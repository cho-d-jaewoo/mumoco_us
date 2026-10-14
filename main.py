"""Experiment GUI: run a saved task and record physical, language or multimodal corrections.

Flow: connect -> home -> user ID -> (select task) -> select modality -> roll next scenario -> home -> instructions
      -> START: execute -> START: correction -> START: finish -> (speech to text) -> save? -> home -> roll next ...
A modality is done once each error scenario (A-D) has a correction with it, run in random order (each roll draws
one of the scenarios still open); then the participant is asked to fill out the survey and chooses the next modality.
The user ID gives the participant's task (1, 4, 7, ... = the first task of EXPERIMENT_TASKS, 2, 5, 8, ... the
second, 3, 6, 9, ... the third), so the task selection is skipped; only user ID 0 chooses among all tasks.
Correction: physical = guide the robot; language = the robot stands still while the microphone records;
multimodal = guide the robot while the microphone records.
Participants see only the high-level task (EXPERIMENT_TASKS in config.py); the scenarios stay internal. A
scenario counts as done for a modality once its correction is saved (this session only), so one without a saved
correction can be drawn again; completed modalities and tasks cannot be selected.
"""

import random
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
    result = execute_with_correction(robot, task, correct)
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
        completed = set()                                # (short name, error, modality) with a saved correction
        titles = list(EXPERIMENT_TASKS)
        # Each participant does one task, given by the user ID: 1, 4, 7, ... the first task, 2, 5, 8, ... the
        # second, 3, 6, 9, ... the third. User ID 0 chooses among all tasks.
        assigned = titles[(int(user) - 1) % len(titles)] if int(user) else None
        print(f"[INFO] User ID: {user} ({assigned or 'all tasks'})")

        def scenarios(title):                            # letter -> error, for scenarios with a task file
            short, errors = EXPERIMENT_TASKS[title]
            return {letter: error for letter, error in errors.items() if error and find_task_path(f"{short}_{error}")}

        def modality_done(title, modality):
            short = EXPERIMENT_TASKS[title][0]
            return all((short, error, modality) in completed for error in scenarios(title).values())

        def task_done(title):
            return all(modality_done(title, m) for m in MODALITIES)

        while True:
            done = [i for i, title in enumerate(titles) if task_done(title)]
            if task_done(assigned) if assigned else len(done) == len(titles):
                ui.status("All Tasks Completed", "Thank you for participating!")
                return
            title = assigned or titles[ui.choose("Select Task", titles, completed=done)]
            short = EXPERIMENT_TASKS[title][0]
            choice = ui.choose_modality("Choose Correction Modality", title, [MODALITY_NAMES[m] for m in MODALITIES],
                                        completed=[j for j, m in enumerate(MODALITIES) if modality_done(title, m)],
                                        example=f"{short}_answer")
            if choice is None:
                continue
            modality = MODALITIES[choice]
            name = MODALITY_NAMES[modality]
            while True:                                  # every scenario once with this modality, in random order
                ready = scenarios(title)
                remaining = [letter for letter, error in ready.items() if (short, error, modality) not in completed]
                if not remaining:
                    ui.notify(f"{name} Completed", "You have completed all scenarios with this correction modality.\n\n"
                                                   "Please fill out the survey.")
                    break
                if ui.roll_scenario(name, title, len(ready) - len(remaining), len(ready)) is None:
                    break
                letter = random.choice(remaining)
                error = ready[letter]
                task = load_task(find_task_path(f"{short}_{error}"))
                print(f"[INFO] {title}, {name}: scenario {letter} = {task['name']} (drawn from {', '.join(remaining)})")
                ui.status("Preparing Task", "Returning to home position...", "moving")
                robot.go_home()
                ui.wait_for_start("Instructions", INSTRUCTIONS[name].format(task=title))
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
