from pathlib import Path

import numpy as np

# ---------------- files ----------------
ROOT = Path(__file__).resolve().parent.parent   # repository root
TASK_DIR = ROOT / "tasks"                 # tasks shown in the experiment
ANSWER_DIR = ROOT / "task_answers"        # <base>_answer.json: how each task should be done
VIDEO_DIR = ROOT / "task_videos"          # <task>.webp, made by sim_task.py; answers shown by "Task Example"
CORRECTION_DIR = ROOT / "corrections"
# corrections/<task>/<folder>/ per modality; language and multimodal keep the audio in <folder>/recordings/
CORRECTION_FOLDERS = {"physical": "physical_only", "language": "language_only", "multimodal": "multimodal"}
MODALITY_NAMES = {"physical": "Physical-Only", "language": "Language-Only", "multimodal": "Multimodal"}  # GUI order

# ---------------- experiment tasks (main.py) ----------------
# High-level task -> (short name, error scenarios). Participants only see the letters; each scenario name is
# the error part of its task file, tasks/<short>_<error>.json. None = not implemented yet (shown, not selectable).
EXPERIMENT_TASKS = {
    "Pick and Place": ("pnp", {"A": "spill", "B": "too_high", "C": "place_on_toaster", "D": None}),
    "Bread in Toaster": ("bit", {"A": "off_the_slot", "B": "far_from_user", "C": "to_the_plate", "D": None}),
    "Wipe the Plate": ("wtp", {"A": "lose_contact", "B": "wipe_with_sponge", "C": "no_cloth", "D": None}),
}

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
MOTION_SPEED = 0.05          # [rad/s] a joint faster than this over MOTION_WINDOW counts as the user guiding;
MOTION_WINDOW = 0.2          # [s]     slower is the drift of the compliant robot before/after (measured < 0.01 rad/s)

# ---------------- joystick (joystick_example.py, SteelSeries duo) ----------------
JOY_BUTTONS = {"A": 0, "B": 1, "X": 2, "Y": 3, "BACK": 6, "START": 7}
JOY_HAT = 0                  # D-pad as hat 0 (not in joystick_example.py -> check with `python3 -m mumoco.joystick_input`)
JOY_NAV_AXIS = 1             # left stick up-down (joystick_example.py), also moves menu selections
JOY_NAV_AXIS_X = 0           # left stick left-right (joystick_example.py)
JOY_NAV_THRESHOLD = 0.5

# ---------------- microphone / speech-to-text (deploy.py VoiceTaskRecorder) ----------------
MIC_NAME = "JLAB TALK"       # part of the microphone name (`python3 -m sounddevice` lists the devices)
MIC_SAMPLE_RATE = 16_000     # [Hz] Whisper's input rate
WHISPER_MODEL = "turbo"      # deploy.py default (GPU); small.en / base.en for lower latency or CPU
WHISPER_LANGUAGE = "en"

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
# bit: toaster and plate are placed from the answer task; the bread stands where each task grasps it.
# pnp: the mug stands where each task grasps it (the toaster position is tuned to those cup heights);
#      the dish is where the answer task puts the mug down.
# wtp: cloth where the answer task grasps it, sponge where wtp_wipe_with_sponge grasps it.
# Camera: side view with the robot on the left and the task area on the right.
SCENE_CAMERAS = {
    "pnp": {"cameraDistance": 1.55, "cameraYaw": 5.0, "cameraPitch": -12.0, "cameraTargetPosition": [0.45, 0.0, 0.32]},
    "wtp": {"cameraDistance": 1.35, "cameraYaw": -25.0, "cameraPitch": -40.0, "cameraTargetPosition": [0.5, 0.0, 0.2]},   # higher, whole arm in view
    "bit": {"cameraDistance": 1.45, "cameraYaw": -22.0, "cameraPitch": -40.0, "cameraTargetPosition": [0.46, 0.0, 0.24]},   # higher, from the robot side: toaster does not hide the plate
}
PNP_CUP_SCALE = 1.2          # pybullet_data mug; it stands where the gripper first closes, handle toward the robot
PNP_TOASTER = {"x": 0.65, "y": -0.035, "scale": 1.0}    # original size, midway between cup and goal:
                                                         # answer/spill pass above it, place_on_toaster puts the mug on it
PNP_DISH = {"radius": 0.09, "thickness": 0.008}          # blue dish under the mug where pnp_answer puts it down
BIT_PLATE = {"x": 0.71, "y": 0.20, "radius": 0.09}   # blue plate where bit_to_the_plate releases the bread
WTP_PLATE = {"x": 0.69, "y": -0.01, "radius": 0.12}  # white plate under the area wtp_answer wipes
WTP_SPONGE = {"task": "wtp_wipe_with_sponge",        # sponge stands where this task grasps it
              "size": [0.10, 0.07, 0.06], "pad": 0.012}  # [m] length, width (across the fingers), height; scour pad below
