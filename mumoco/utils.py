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

from .config import *
from .franka import Franka
from .trajectory_utils import JointSpline, active_interval, task_segments


class CorrectionRequested(Exception):
    """Raised inside a motion after a correction was requested during task execution."""


class Stopped(Exception):
    """Raised inside robot operations when the application is shutting down."""


class Robot:
    """Arm + gripper connection with the proven tutorial.py motion primitives."""

    def __init__(self, should_stop=lambda: False):
        self.franka = Franka()
        self.home = self.franka.home
        self.should_stop = should_stop     # checked in every waiting loop -> raises Stopped
        self.gripper_open = None           # measured is_open, streamed by gripper_control_cjw
        self.gripper_target = None         # pending non-blocking command (request_gripper)
        self.gripper_deadline = 0.0
        self.in_correction = False         # NUC ignores send2robot while in correction mode
        self.correction_requested = False  # set by Ctrl+C, the GUI or the joystick; checked by move_joint

        print("[INFO] Waiting for arm_control_cjw and gripper_control_cjw on the NUC...")
        self.conn, self.grip = self.franka.connect_robot_and_gripper(ARM_PORT, GRIPPER_PORT, self._check_stop)
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

    def _check_stop(self):
        if self.should_stop():
            raise Stopped

    # ---------------- motion ----------------
    def move_joint(self, q_goal):
        """tutorial.move_joint: joint-space P-control at HZ, all joints arrive together.

        Also accepts the goal if the error stops shrinking for STALL_TIME while within STALL_TOL
        (the compliant robot, a payload or contact can leave a small offset the P-control never removes).
        """
        q_goal = np.asarray(q_goal, dtype=np.float64)
        start = time.monotonic()
        best, best_time = np.inf, start
        while True:
            self._check_stop()
            if self.correction_requested:
                raise CorrectionRequested
            t0 = time.monotonic()
            err = q_goal - self.franka.readState(self.conn)["q"]
            e = np.max(np.abs(err))
            if e < JOINT_TOL:
                break
            if e < best - 1e-3:
                best, best_time = e, t0
            elif e < STALL_TOL and t0 - best_time > STALL_TIME:
                print(f"[WARNING] Target reached within {e:.3f} rad (error stopped shrinking).")
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
            self._check_stop()
            select.select([self.grip], [], [], 0.05)

    def gripper_busy(self):
        """True while a request_gripper command has not reported its new state yet."""
        if self.gripper_target is not None and (self.gripper_open == self.gripper_target
                                                or time.monotonic() > self.gripper_deadline):
            self.gripper_target = None
        return self.gripper_target is not None

    def request_gripper(self, open_):
        """Send an open/close command without waiting (guiding mode keeps recording meanwhile).
        Ignored if the gripper already is in that state or still runs the previous command
        (gripper_control_cjw would drop a queued command). Returns True if sent."""
        self.update_gripper()
        if self.gripper_busy() or open_ == self.gripper_open:
            return False
        self.franka.send2gripper(self.grip, "o" if open_ else "c")
        self.gripper_target, self.gripper_deadline = open_, time.monotonic() + GRIPPER_TIMEOUT
        return True

    def go_home(self):
        """Arm to home first, then open the gripper."""
        print("[INFO] Returning robot to home position...")
        self.move_joint(self.home)
        self.set_gripper(True)
        print("[INFO] Robot returned to home position.")

    def follow(self, points):
        """Move smoothly through `points` without stopping (JointSpline from the current configuration,
        velocity feed-forward + P feedback at HZ), then settle on the last point with move_joint."""
        spline = JointSpline([self.franka.readState(self.conn)["q"], *points], VMAX, AMAX)
        start = time.monotonic()
        while (t := time.monotonic() - start) < spline.duration:
            self._check_stop()
            if self.correction_requested:
                raise CorrectionRequested
            q_ref, qd_ref = spline.sample(t + 1 / HZ)      # aim one command period ahead
            qdot = qd_ref + GAIN * (q_ref - self.franka.readState(self.conn)["q"])
            self.franka.send2robot(self.conn, np.clip(qdot, -2 * VMAX, 2 * VMAX))
            time.sleep(max(0.0, 1 / HZ - (time.monotonic() - start - t)))
        self.move_joint(points[-1])

    def execute_task(self, waypoints):
        """Drive through the waypoints without stopping; stop only where the gripper has to change
        (then set it and wait for it) and at the last waypoint (see task_segments)."""
        for points, gripper_open, last in task_segments(waypoints, self.update_gripper()):
            print(f"[INFO] Moving to waypoint {last}/{len(waypoints)}")
            self.follow(points)
            self.set_gripper(gripper_open)
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

    def guide(self, record, tick):
        """Correction mode of arm_control_cjw: "c" -> operator guides the robot -> "v".

        Calls record(t, state, gripper_open) for EVERY arm state message the NUC streams
        (franka.listen2robot would keep only the newest). state = {"q", "O_F", "J"} as in franka.py;
        gripper_open is the newest measured gripper state when that arm message is read.
        tick() is called after every socket read (must not block); guiding ends when it returns True.
        """
        self.franka.send2mode(self.conn, "c")
        self.in_correction = True
        print("\n[CORRECTION] Correction mode active.")
        time.sleep(SETTLE_TIME)                         # let an interrupted motion ramp down
        self.franka.readState(self.conn)                # drop states received before recording
        self.update_gripper()
        buf, t0 = "", time.monotonic()
        while True:
            self._check_stop()
            ready = select.select([self.conn, self.grip], [], [], 0.1)[0]
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
            if tick():
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


# ---------------- terminal input (record_tasks.py) ----------------
def ask_yes_no(prompt):
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def read_terminal_line():
    """A typed line (stripped, lower case) if one is waiting, else None (never blocks)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


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


def find_task_path(name):
    """tasks/<name>.json or task_answers/<name>.json, or None."""
    return next((path for path in (TASK_DIR / f"{name}.json", ANSWER_DIR / f"{name}.json") if path.exists()), None)


def answer_task_name(task_name):
    """'<base>_<error type>' -> '<base>_answer' (e.g. pnp_straight_to_goal -> pnp_answer).

    The base is the longest one with an answer file in task_answers/ that prefixes the task name
    (error types may contain '_'); without a matching file, the base is the first word of the name.
    """
    bases = sorted((path.stem[:-len("_answer")] for path in ANSWER_DIR.glob("*_answer.json")), key=len, reverse=True)
    base = next((b for b in bases if task_name == f"{b}_answer" or task_name.startswith(f"{b}_")),
                task_name.split("_")[0])
    return f"{base}_answer"


def task_base_name(task_name):
    return answer_task_name(task_name)[:-len("_answer")]


def task_video_path(task_name):
    return VIDEO_DIR / f"{task_name}.webp"


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
def execute_with_physical_correction(robot, waypoints, tick):
    """Run the task; once robot.correction_requested is set (Ctrl+C, GUI, joystick) switch to
    correction mode and return every recorded (t, state, gripper_open) sample.
    Returns None if the task finished without correction. tick: see Robot.guide."""
    samples = []
    try:
        robot.execute_task(waypoints)
        return None
    except CorrectionRequested:
        print("\n[INFO] Correction requested. Task execution stopped.")
        robot.guide(lambda t, state, gripper_open: samples.append((t, state, gripper_open)), tick)
        return samples
    finally:
        robot.correction_requested = False


def save_correction(task_name, modality, samples):
    """Slice the inactive head/tail and save the dense correction trajectory; returns the path (None if nothing)."""
    interval = active_interval([s["q"] for _, s, _ in samples], [g for _, _, g in samples])
    if interval is None:
        print("[WARNING] No motion or gripper action detected. Nothing saved.")
        return None
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
    return path
