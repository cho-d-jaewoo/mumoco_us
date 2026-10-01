from pathlib import Path

import numpy as np

# ---------------- files ----------------
ROOT = Path(__file__).resolve().parent.parent   # repository root
TASK_DIR = ROOT / "tasks"                 # tasks shown in the experiment
ANSWER_DIR = ROOT / "task_answers"        # <base>_answer.json: how each task should be done
VIDEO_DIR = ROOT / "task_videos"          # <task>.webp, made by sim_task.py, shown by "View Scenario"
CORRECTION_DIR = ROOT / "corrections"

# ---------------- connection (arm_control_cjw.cpp / gripper_control_cjw.cpp) ----------------
ARM_PORT = 8080
GRIPPER_PORT = 8081
STATE_LENGTH = 55            # q[7] + O_F[6] + J[42] per state message (franka.listen2robot)
STATE_HZ = 100               # state_frequency argument of arm_control_cjw (default 100)
CONNECT_TIMEOUT = 5.0        # [s] first state must arrive within this time

# ---------------- motion (tutorial.move_joint / tutorial.gripper) ----------------
HZ = 20                      # command rate = policy_frequency on the NUC
VMAX = 0.3                   # [rad/s] largest joint speed
GAIN = 1.5                   # P gain in joint space
JOINT_TOL = 0.01             # [rad] waypoint (and home) reached
MOVE_TIMEOUT = 20.0          # [s]
STALL_TIME = 1.0             # [s] error no longer shrinking for this long ...
STALL_TOL = 0.05             # [rad] ... while within this -> accept the waypoint (compliant robot, contact)
AMAX = 1.0                   # [rad/s^2] largest joint acceleration of the smooth task trajectory
SETTLE_TIME = 0.3            # [s] after a stop command
MODE_SWITCH_TIME = 0.5       # [s] after sending "v"
GRIPPER_TIMEOUT = 5.0        # [s] gripper must report the new state (= motion finished) within this time
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # Panda joint limits [rad]
UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# ---------------- trajectory slicing ----------------
MOTION_THRESHOLD = 1e-4      # [rad] joint change between consecutive state messages that counts as motion
                             # (1e-4 rad per message at 100 Hz = 0.01 rad/s)

# ---------------- joystick (joystick_example.py, SteelSeries duo) ----------------
JOY_BUTTONS = {"A": 0, "B": 1, "X": 2, "Y": 3, "BACK": 6, "START": 7}
JOY_HAT = 0                  # D-pad as hat 0 (not in joystick_example.py -> check with `python3 -m mumoco.joystick_input`)
JOY_NAV_AXIS = 1             # left stick up-down (joystick_example.py), also moves menu selections
JOY_NAV_AXIS_X = 0           # left stick left-right (joystick_example.py)
JOY_NAV_THRESHOLD = 0.5

# ---------------- PyBullet simulation (sim_task.py, sim_correction.py) ----------------
SIM_DT = 1 / 240             # [s] PyBullet time step (panda-tutorial)
SIM_GRIPPER_TIME = 1.0       # [s] simulated gripper open/close
SIM_ARM_FORCES = [87, 87, 87, 87, 12, 12, 12]   # [Nm] Panda joint torque limits (lfc_test)
SIM_CAMERA = {"cameraDistance": 1.6, "cameraYaw": 50.0, "cameraPitch": -30.0, "cameraTargetPosition": [0.4, 0.0, 0.3]}

# ---------------- task videos (sim_task.py) ----------------
VIDEO_FPS = 20
VIDEO_SIZE = (640, 360)      # [px] width, height
VIDEO_HOLD = (0.8, 1.5)      # [s] still frames before the motion starts and after it ends

# ---------------- simulation scenes, chosen by the task base name (pnp_spill -> "pnp") ----------------
# bit: objects are placed from the answer task, so every bit task starts from the same scene.
# pnp: the mug stands where each task grasps it (the toaster position is tuned to those cup heights).
# Camera: side view with the robot on the left and the task area on the right.
SCENE_CAMERAS = {
    "pnp": {"cameraDistance": 1.55, "cameraYaw": 5.0, "cameraPitch": -12.0, "cameraTargetPosition": [0.45, 0.0, 0.32]},
    "bit": {"cameraDistance": 1.45, "cameraYaw": -22.0, "cameraPitch": -40.0, "cameraTargetPosition": [0.46, 0.0, 0.24]},   # higher, from the robot side: toaster does not hide the plate
}
PNP_CUP_SCALE = 1.2          # pybullet_data mug; it stands where the gripper first closes, handle toward the robot
PNP_TOASTER = {"x": 0.65, "y": -0.035, "scale": 1.0}    # original size, midway between cup and goal:
                                                         # answer/spill pass above it, down_too_early/straight_to_goal hit it
BIT_PLATE = {"x": 0.71, "y": 0.20, "radius": 0.09}   # blue plate where bit_to_the_plate releases the bread
