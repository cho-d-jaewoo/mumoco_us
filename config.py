from pathlib import Path

import numpy as np

# ---------------- files ----------------
ROOT = Path(__file__).resolve().parent
TASK_DIR = ROOT / "tasks"
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
SETTLE_TIME = 0.3            # [s] after a stop command
MODE_SWITCH_TIME = 0.5       # [s] after sending "v"
GRIPPER_TIMEOUT = 5.0        # [s] gripper must report the new state (= motion finished) within this time
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # Panda joint limits [rad]
UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

# ---------------- trajectory slicing ----------------
MOTION_THRESHOLD = 1e-4      # [rad] joint change between consecutive state messages that counts as motion
                             # (1e-4 rad per message at 100 Hz = 0.01 rad/s)
