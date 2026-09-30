"""Replay a recorded task in PyBullet.

    python3 sim_task.py test
"""

import argparse

from mumoco.config import SIM_TASK_COLOR, TASK_DIR
from mumoco.sim import Simulation, plan_task, play_task
from mumoco.utils import load_task


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded task in PyBullet.")
    parser.add_argument("task", help="task name in tasks/ (without .json)")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default: real time)")
    parser.add_argument("--no-gui", action="store_true", help="run without a window (quick check)")
    args = parser.parse_args()
    path = TASK_DIR / f"{args.task}.json"
    if not path.exists():
        parser.error(f"no task '{args.task}'. Available: {', '.join(p.stem for p in sorted(TASK_DIR.glob('*.json')))}")
    task = load_task(path)

    sim = Simulation(gui=not args.no_gui, speed=args.speed)
    sim.mark_waypoints(task["waypoints"])
    sim.show(f"Following task: {task['name']}", SIM_TASK_COLOR)
    play_task(sim, plan_task(task["waypoints"]))
    sim.show("Task finished", SIM_TASK_COLOR)
    sim.hold()


if __name__ == "__main__":
    main()
