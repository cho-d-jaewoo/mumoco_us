"""Make the demonstration video of a task: the robot executes the task once in PyBullet.

    python3 sim_task.py pnp_spill          # -> task_videos/pnp_spill.webp
    python3 sim_task.py --all              # every task in tasks/ and task_answers/
    python3 sim_task.py pnp_spill --show   # watch it in a PyBullet window instead (no video)
"""

import argparse

from mumoco.config import ANSWER_DIR, TASK_DIR, VIDEO_HOLD
from mumoco.sim import make_simulation, plan_task, play_task, save_video
from mumoco.utils import find_task_path, load_task, task_base_name, task_video_path


def make_video(name, show=False, speed=1.0):
    task = load_task(find_task_path(name))
    sim = make_simulation(task_base_name(name), task["waypoints"], gui=show, speed=speed)
    if not show:
        sim.record()
    sim.wait(VIDEO_HOLD[0])                        # start pose
    play_task(sim, plan_task(task["waypoints"]))
    sim.wait(VIDEO_HOLD[1])                        # end pose, released object settles
    if show:
        sim.hold()
    else:
        path = task_video_path(name)
        save_video(sim.frames, path)
        print(f"[INFO] {path.relative_to(path.parent.parent)}: {len(sim.frames)} frames")
    import pybullet
    pybullet.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Make the PyBullet demonstration video of a task.")
    parser.add_argument("task", nargs="?", help="task name in tasks/ or task_answers/ (without .json)")
    parser.add_argument("--all", action="store_true", help="every task in tasks/ and task_answers/")
    parser.add_argument("--show", action="store_true", help="watch in a PyBullet window instead of saving a video")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed (default: real time)")
    args = parser.parse_args()
    available = sorted(path.stem for path in [*TASK_DIR.glob("*.json"), *ANSWER_DIR.glob("*.json")])
    names = available if args.all else [args.task]
    if not args.all and (args.task is None or find_task_path(args.task) is None):
        parser.error(f"give a task or --all. Available: {', '.join(available)}")
    for name in names:
        make_video(name, show=args.show, speed=args.speed)


if __name__ == "__main__":
    main()
