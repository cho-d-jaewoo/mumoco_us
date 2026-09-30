"""Robot session, task files and correction files.
"""

import json
import re
import select
import signal
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime

import numpy as np

from config import *
from franka import Franka
from trajectory_utils import active_interval


class CorrectionRequested(Exception):
    """Raised inside a motion after Ctrl+C was pressed during task execution."""


class Robot:
    """Arm + gripper connection with the proven tutorial.py motion primitives."""

    def __init__(self):
        self.franka = Franka()
        self.home = self.franka.home
        self.gripper_open = None           # measured is_open, streamed by gripper_control_cjw
        self.in_correction = False         # NUC ignores send2robot while in correction mode
        self.correction_requested = False

        print("[INFO] Waiting for arm_control_cjw and gripper_control_cjw on the NUC...")
        self.conn, self.grip = self.franka.connect_robot_and_gripper(ARM_PORT, GRIPPER_PORT)
        self.conn.settimeout(CONNECT_TIMEOUT)
        try:
            q = self.franka.readState(self.conn)["q"]
        except socket.timeout:
            raise RuntimeError("Arm controller connected but sends no state.") from None
        self.conn.settimeout(None)
        if not np.all(np.isfinite(q)):
            raise RuntimeError(f"Invalid joint state: {q}")
        print("[INFO] Arm connected successfully.")
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while self.update_gripper() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("Gripper controller connected but sends no state (gripper_control_cjw?).")
            time.sleep(0.05)
        print("[INFO] Gripper connected successfully.")

    # ---------------- motion ----------------
    def move_joint(self, q_goal):
        """tutorial.move_joint: joint-space P-control at HZ, all joints arrive together."""
        q_goal = np.asarray(q_goal, dtype=np.float64)
        start = time.monotonic()
        while True:
            if self.correction_requested:
                raise CorrectionRequested
            t0 = time.monotonic()
            err = q_goal - self.franka.readState(self.conn)["q"]
            if np.max(np.abs(err)) < JOINT_TOL:
                break
            if t0 - start > MOVE_TIMEOUT:
                raise TimeoutError(f"Joint error still {np.round(err, 3)}")
            qdot = GAIN * err
            qdot *= min(1.0, VMAX / np.max(np.abs(qdot)))  # straight line in joint space
            self.franka.send2robot(self.conn, qdot)
            time.sleep(max(0.0, 1 / HZ - (time.monotonic() - t0)))
        self.franka.send2robot(self.conn, np.zeros(7))
        time.sleep(SETTLE_TIME)

    def update_gripper(self):
        """Drain the gripper socket ("s,<0|1>," messages) and keep the newest is_open."""
        data = b""
        while select.select([self.grip], [], [], 0)[0]:
            chunk = self.grip.recv(4096)
            if not chunk:
                raise ConnectionError("Gripper controller closed the connection.")
            data += chunk
        values = re.findall(r"s,([01]),", data.decode(errors="ignore"))  # partial messages are skipped
        if values:
            self.gripper_open = values[-1] == "1"
        return self.gripper_open

    def set_gripper(self, open_):
        """Open (True) or close (False) the gripper and block until it has finished; skipped if it already is.

        gripper_control_cjw streams nothing while move()/grasp() runs, so the first message
        with the new state arrives only after the gripper motion is done.
        """
        if open_ == self.update_gripper():
            return
        self.franka.send2gripper(self.grip, "o" if open_ else "c")
        deadline = time.monotonic() + GRIPPER_TIMEOUT
        while self.update_gripper() != open_:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Gripper did not {'open' if open_ else 'close'} within {GRIPPER_TIMEOUT} s.")
            select.select([self.grip], [], [], 0.05)

    def go_home(self):
        """Arm to home first, then open the gripper."""
        print("[INFO] Returning robot to home position...")
        self.move_joint(self.home)
        self.set_gripper(True)
        print("[INFO] Robot returned to home position.")

    def execute_task(self, waypoints):
        """Per waypoint: move the arm there (and stop), then set the gripper and wait for it."""
        for i, wp in enumerate(waypoints, 1):
            print(f"[INFO] Waypoint {i}/{len(waypoints)}")
            self.move_joint(wp["joint_positions"])
            self.set_gripper(wp["gripper_open"])
            if self.correction_requested:
                raise CorrectionRequested

    # ---------------- correction (hand-guiding) mode ----------------
    @contextmanager
    def ctrl_c_requests_correction(self):
        """Inside this block the 1st Ctrl+C requests correction mode and the 2nd quits (tutorial.py)."""
        def on_ctrl_c(sig, frame):
            if self.correction_requested:
                raise KeyboardInterrupt
            self.correction_requested = True           # only a flag -> never cuts a message in half
        previous = signal.signal(signal.SIGINT, on_ctrl_c)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous)
            self.correction_requested = False

    def wait_for_ctrl_c(self):
        while not self.correction_requested:
            time.sleep(0.05)

    def guide(self, record):
        """Correction mode of arm_control_cjw: "c" -> operator guides the robot -> "v".

        Calls record(t, state, gripper_open) for EVERY arm state message the NUC streams
        (franka.listen2robot would keep only the newest). state = {"q", "O_F", "J"} as in franka.py;
        gripper_open is the newest measured gripper state when that arm message is read.
        """
        self.franka.send2mode(self.conn, "c")
        self.in_correction = True
        print("\n[CORRECTION] Activate the External Activation Switch and guide the robot"
              " (open/close the gripper directly as needed).")
        print("[CORRECTION] Finish: deactivate the switch, then press Enter. (Ctrl+C: quit)")
        time.sleep(SETTLE_TIME)                         # let an interrupted motion ramp down
        self.franka.readState(self.conn)                # drop states received before recording
        self.update_gripper()
        buf, t0 = "", time.monotonic()
        while True:
            ready = select.select([self.conn, self.grip, sys.stdin], [], [], 0.1)[0]
            if self.grip in ready:                      # gripper first -> arm samples get the newest state
                self.update_gripper()
            if self.conn in ready:
                data = self.conn.recv(65536)
                if not data:
                    raise ConnectionError("Arm controller closed the connection.")
                *messages, buf = (buf + data.decode(errors="ignore")).split("s,")
                t = time.monotonic() - t0
                for m in messages:
                    v = m.split(",")[:-1]
                    if len(v) == STATE_LENGTH:          # truncated messages are skipped
                        v = np.asarray(v, dtype=np.float64)
                        record(t, {"q": v[:7], "O_F": v[7:13], "J": v[13:].reshape((7, 6)).T},
                               self.gripper_open)
            if sys.stdin in ready:
                sys.stdin.readline()
                break
        self.franka.send2mode(self.conn, "v")
        time.sleep(MODE_SWITCH_TIME)
        self.in_correction = False
        print("[CORRECTION] Done. Robot is back in velocity control.")

    def close(self):
        if not self.in_correction:                       # in correction mode the NUC only accepts 'v'/'c'
            try:
                self.franka.send2robot(self.conn, np.zeros(7))
            except OSError:
                pass
        self.conn.close()
        self.grip.close()
        print("[INFO] Robot connections closed.")


# ---------------- terminal input ----------------
def ask_yes_no(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def choose(title, options, allow_back="q"):
    """Numbered menu; returns the chosen index, or None for `allow_back`."""
    print(f"\n{title}\n")
    for i, option in enumerate(options, 1):
        print(f"[{i}] {option}")
    while True:
        answer = input(f"\nSelect (1-{len(options)}, {allow_back} = back): ").strip().lower()
        if answer == allow_back:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("[WARNING] Invalid selection.")


# ---------------- tasks ----------------
def load_task(path):
    """Read and validate a task file; raises ValueError if malformed."""
    data = json.loads(path.read_text())
    waypoints = []
    for wp in data["waypoints"]:
        q = np.asarray(wp["joint_positions"], dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)) or np.any(q < LOWER) or np.any(q > UPPER):
            raise ValueError(f"invalid joint positions {wp['joint_positions']}")
        if not isinstance(wp["gripper_open"], bool):
            raise ValueError("gripper_open must be true or false")
        waypoints.append({"joint_positions": q, "gripper_open": wp["gripper_open"]})
    if not waypoints:
        raise ValueError("no waypoints")
    return {"name": path.stem, "waypoints": waypoints}


def load_tasks():
    tasks = []
    for path in sorted(TASK_DIR.glob("*.json")):
        try:
            tasks.append(load_task(path))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"[WARNING] Skipping malformed task file {path.name}: {e}")
    return tasks


def ask_task_name():
    while True:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", input("Task name: ").strip()).strip("_")
        if not name:
            print("[WARNING] Use letters, digits, '_' or '-'.")
        elif not (TASK_DIR / f"{name}.json").exists() or ask_yes_no(f"Task '{name}' exists. Overwrite?"):
            return name


def save_task(name, waypoints):
    """Save the compact waypoint representation (one waypoint per line)."""
    TASK_DIR.mkdir(exist_ok=True)
    lines = ",\n    ".join(json.dumps({"joint_positions": np.round(wp["joint_positions"], 6).tolist(),
                                       "gripper_open": bool(wp["gripper_open"])}) for wp in waypoints)
    path = TASK_DIR / f"{name}.json"
    path.write_text(f'{{\n  "name": {json.dumps(name)},\n  "waypoints": [\n    {lines}\n  ]\n}}\n')
    print(f"[INFO] Task saved: {path.relative_to(ROOT)}")


# ---------------- physical corrections (dense) ----------------
def execute_with_physical_correction(robot, waypoints):
    """Run the task; on Ctrl+C switch to correction mode and return every recorded
    (t, state, gripper_open) sample. Returns None if the task finished without correction."""
    samples = []
    with robot.ctrl_c_requests_correction():
        try:
            robot.execute_task(waypoints)
            return None
        except CorrectionRequested:
            print("\n[INFO] Correction requested. Task execution stopped.")
            robot.guide(lambda t, state, gripper_open: samples.append((t, state, gripper_open)))
    return samples


def save_correction(task_name, modality, samples):
    """Slice the inactive head/tail and save the dense correction trajectory."""
    interval = active_interval([s["q"] for _, s, _ in samples], [g for _, _, g in samples])
    if interval is None:
        print("[WARNING] No motion or gripper action detected. Nothing saved.")
        return
    start, end = interval
    t_start = samples[start][0]
    trajectory = [{"t": round(t - t_start, 4), "q": s["q"].tolist(), "O_F": s["O_F"].tolist(),
                   "J": s["J"].tolist(), "gripper_open": g} for t, s, g in samples[start:end + 1]]
    now = datetime.now()
    data = {
        "task_name": task_name,
        "modality": modality,
        "recorded_at": now.isoformat(timespec="seconds"),
        "num_recorded_samples": len(samples),
        "num_samples": len(trajectory),
        "state_rate_hz": STATE_HZ,
        "fields": {
            "t": "[s] host receive time since the first saved sample (messages carry no NUC timestamp)",
            "q": "[rad] measured joint positions (robot_state.q)",
            "O_F": "estimated external wrench in base frame (robot_state.O_F_ext_hat_K)",
            "J": "6x7 zero Jacobian at the end effector",
            "gripper_open": "measured gripper state (width > 0.0725 m), newest value when the arm state was read",
        },
        "trajectory": trajectory,
    }
    folder = CORRECTION_DIR / task_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"correction_{now:%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(data))
    print(f"[INFO] Correction saved: {path.relative_to(ROOT)} "
          f"({len(trajectory)} of {len(samples)} samples, {trajectory[-1]['t']:.1f} s)")
