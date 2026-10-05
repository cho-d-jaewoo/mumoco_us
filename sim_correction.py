"""Replay a task in PyBullet until the moment a recorded physical correction began, then replay the correction.

    python3 sim_correction.py pnp_spill               # newest physical / multimodal correction of task 'pnp_spill'
    python3 sim_correction.py pnp_spill --correction corrections/pnp_spill/physical_only/spill_1234_physical_20261001_153012.json
"""

import argparse
from pathlib import Path

from mumoco.sim import (find_correction_start, load_correction, make_simulation, newest_correction, plan_task,
                        play_correction, play_task)
from mumoco.utils import find_task_path, load_task, task_base_name


def main():
    parser = argparse.ArgumentParser(description="Replay a task and a recorded physical correction in PyBullet.")
    parser.add_argument("task", help="task name in tasks/ (without .json)")
    parser.add_argument("--correction", type=Path, help="correction file (default: newest one of the task)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default: real time)")
    parser.add_argument("--no-gui", action="store_true", help="run without a window (quick check)")
    args = parser.parse_args()
    path = find_task_path(args.task)
    if path is None:
        parser.error(f"no task '{args.task}' in tasks/ or task_answers/")
    task = load_task(path)
    correction_path = args.correction or newest_correction(args.task)
    if correction_path is None:
        parser.error(f"no physical or multimodal corrections for task '{args.task}' in corrections/{args.task}/")
    correction = load_correction(correction_path)
    if correction["task_name"] != args.task:
        print(f"[WARNING] The correction was recorded for task '{correction['task_name']}'.")

    plan = plan_task(task["waypoints"])
    first = correction["trajectory"][0]
    segment, t, distance = find_correction_start(plan, first["q"], first["gripper_open"])
    print(f"[INFO] Correction: {correction_path.name} ({len(correction['trajectory'])} samples)")
    print(f"[INFO] It began on the way to waypoint {plan[segment][2]} "
          f"({distance:.3f} rad from the planned path).")
    if distance > 0.1:
        print("[WARNING] The correction does not start on the task path; the robot jumps to its start.")

    sim = make_simulation(task_base_name(args.task), task["waypoints"], gui=not args.no_gui, speed=args.speed)
    print(f"[SIM] Following task: {task['name']}")
    play_task(sim, plan, stop=(segment, t))
    print("[SIM] Physical correction (recorded)")
    play_correction(sim, correction)
    print("[SIM] Correction finished")
    sim.hold()


if __name__ == "__main__":
    main()
