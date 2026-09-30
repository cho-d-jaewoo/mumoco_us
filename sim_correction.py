"""Replay a task in PyBullet until the moment a recorded physical correction began, then replay the correction.

    python3 sim_correction.py test                    # newest correction of task 'test'
    python3 sim_correction.py test --correction corrections/test/correction_20260930_150108.json
"""

import argparse
from pathlib import Path

from mumoco.config import SIM_CORRECTION_COLOR, SIM_TASK_COLOR, TASK_DIR
from mumoco.sim import (Simulation, find_correction_start, load_correction, newest_correction, plan_task,
                        play_correction, play_task)
from mumoco.utils import load_task


def main():
    parser = argparse.ArgumentParser(description="Replay a task and a recorded physical correction in PyBullet.")
    parser.add_argument("task", help="task name in tasks/ (without .json)")
    parser.add_argument("--correction", type=Path, help="correction file (default: newest one of the task)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default: real time)")
    parser.add_argument("--no-gui", action="store_true", help="run without a window (quick check)")
    args = parser.parse_args()
    path = TASK_DIR / f"{args.task}.json"
    if not path.exists():
        parser.error(f"no task '{args.task}'. Available: {', '.join(p.stem for p in sorted(TASK_DIR.glob('*.json')))}")
    task = load_task(path)
    correction_path = args.correction or newest_correction(args.task)
    if correction_path is None:
        parser.error(f"no corrections for task '{args.task}' in corrections/{args.task}/")
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

    sim = Simulation(gui=not args.no_gui, speed=args.speed)
    sim.mark_waypoints(task["waypoints"])
    sim.show(f"Following task: {task['name']}", SIM_TASK_COLOR)
    play_task(sim, plan, stop=(segment, t))
    sim.show("Physical correction (recorded)", SIM_CORRECTION_COLOR)
    play_correction(sim, correction)
    sim.show("Correction finished", SIM_CORRECTION_COLOR)
    sim.hold()


if __name__ == "__main__":
    main()
