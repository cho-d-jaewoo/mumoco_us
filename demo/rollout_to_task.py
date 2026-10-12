"""Use a saved policy rollout (demo/rollout.py) as experiment scenario D: copy it to tasks/<task>_policy.json.

    python3 -m demo.rollout_to_task demo/rollouts/pnp/pnp_pnp_v1_9999_20261012_153000.json
    python3 sim_task.py pnp_policy                 # its task video
    python3 dry_run.py                             # scenario D is now selectable (python3 main.py with the robot)

The experiment then replays the recorded trajectory as it happened (Robot.replay), without the policy server.
"""

import argparse
import json
from pathlib import Path

from mumoco.config import EXPERIMENT_TASKS, TASK_DIR
from mumoco.utils import ask_yes_no, load_task


def main():
    parser = argparse.ArgumentParser(description="Make a saved policy rollout the task of experiment scenario D.")
    parser.add_argument("rollout", type=Path, help="demo/rollouts/<task>/<name>.json")
    parser.add_argument("--name", help="task name (default: <task>_<scenario D name in config.py>, e.g. pnp_policy)")
    args = parser.parse_args()
    task = load_task(args.rollout)                              # raises if it is not a valid rollout / task
    if "trajectory" not in task:
        parser.error(f"{args.rollout} has no trajectory (not a rollout saved by demo/rollout.py)")
    data = json.loads(args.rollout.read_text())
    scenario = {short: errors.get("D") for short, errors in EXPERIMENT_TASKS.values()}
    name = args.name or f"{data['task']}_{scenario.get(data['task']) or 'policy'}"
    path = TASK_DIR / f"{name}.json"
    print(f"[INFO] {args.rollout.name}: policy {data['policy']['name']}, success {data['success']}, "
          f"{data['duration']} s, {len(data['trajectory'])} samples")
    if path.exists() and not ask_yes_no(f"tasks/{name}.json exists. Overwrite?"):
        return
    text = args.rollout.read_text().replace(json.dumps(data["name"]), json.dumps(name), 1)   # the "name" entry
    path.write_text(text.replace('"task":', f'"source": {json.dumps(args.rollout.name)},\n  "task":', 1))
    load_task(path)
    print(f"[INFO] Task saved: tasks/{name}.json. Task video: python3 sim_task.py {name}")


if __name__ == "__main__":
    main()
