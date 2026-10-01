"""PyBullet replay of recorded tasks and corrections (scene and Panda as in panda-tutorial / lfc_test).

The task is driven exactly as Robot.execute_task drives the real robot: the same task_segments and
JointSpline, starting at home with the gripper open. The robot follows the reference with position control.
"""

import json
import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from .config import (AMAX, CORRECTION_DIR, PNP_CUP_SCALE, PNP_TOASTER, ROOT, SCENE_CAMERAS, SIM_ARM_FORCES,
                     SIM_CAMERA, SIM_DT, SIM_GRIPPER_TIME, VIDEO_FPS, VIDEO_SIZE, VMAX)
from .franka import Franka
from .trajectory_utils import JointSpline, task_segments

HOME = Franka().home
EE_LINK = 11                        # panda_grasptarget in pybullet_data's panda.urdf
FINGERS = [9, 10]
FINGER_OPEN, FINGER_CLOSED = 0.04, 0.0
MARKER_COLOR = [102 / 255, 102 / 255, 102 / 255]
DATA = pybullet_data.getDataPath()
TABLE_HEIGHT = 0.626                # table.urdf: top surface above its base


class Simulation:
    """Panda next to a table. show() puts a large status label above the robot and draws the end-effector
    trail in its color (sim_correction.py); without show() only the robot and the scene are visible."""

    def __init__(self, gui=True, speed=1.0, camera=SIM_CAMERA):
        self.gui, self.speed, self.camera = gui, speed, camera
        p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(**camera)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_DT)
        self.panda = p.loadURDF(os.path.join(DATA, "franka_panda/panda.urdf"), useFixedBase=True)
        self.graspable, self.held = [], None       # bodies the gripper can pick up / (body, constraint)
        self.label, self.trail, self.color, self.last_ee = None, False, MARKER_COLOR, None
        self.time, self.wall_start, self.gripper_open = 0.0, None, True
        self.frames, self.next_frame = None, 0.0
        self.reset(HOME)

    def add_table(self, top=0.0):
        """Floor and table with its top at height `top` (robot base at 0). A table higher than the robot
        base stands in front of the robot, which then gets its own stand."""
        floor = top - TABLE_HEIGHT
        scenery = [p.loadURDF(os.path.join(DATA, "plane.urdf"), basePosition=[0, 0, floor])]
        if top > 0.0:
            scenery.append(p.loadURDF(os.path.join(DATA, "table/table.urdf"), basePosition=[0.95, 0, floor]))
            stand = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, -floor / 2], rgbaColor=[0.4, 0.4, 0.4, 1])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=stand, basePosition=[0, 0, floor / 2])
        else:                                       # robot on the table (panda-tutorial)
            scenery.append(p.loadURDF(os.path.join(DATA, "table/table.urdf"), basePosition=[0.5, 0, floor]))
        for body in scenery:                        # the recorded motions are real; the table is only for orientation
            self.no_collision(body)

    def hand_pose(self, q):
        """Hand (grasp target) position and orientation at joint positions q, without moving the robot."""
        current, gripper_open = self.q(), self.gripper_open
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.reset(q, gripper_open)
        pose = p.getLinkState(self.panda, EE_LINK, computeForwardKinematics=True)[4:6]
        self.reset(current, gripper_open)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return pose

    def no_collision(self, body):
        for link in range(-1, p.getNumJoints(self.panda)):
            for other in range(-1, p.getNumJoints(body)):
                p.setCollisionFilterPair(self.panda, body, link, other, 0)

    def reset(self, q, gripper_open=True):
        for i, value in enumerate(q):
            p.resetJointState(self.panda, i, value)
        for finger in FINGERS:
            p.resetJointState(self.panda, finger, FINGER_OPEN if gripper_open else FINGER_CLOSED)
        self.command(q)
        self.set_gripper(gripper_open, wait=False)

    def q(self):
        return np.array([state[0] for state in p.getJointStates(self.panda, range(7))])

    def ee(self):
        return np.array(p.getLinkState(self.panda, EE_LINK, computeForwardKinematics=True)[4])

    def command(self, q, qd=None):
        p.setJointMotorControlArray(self.panda, range(7), p.POSITION_CONTROL, targetPositions=list(q),
                                    targetVelocities=list(np.zeros(7) if qd is None else qd),
                                    forces=SIM_ARM_FORCES)

    def set_gripper(self, open_, wait=True):
        """Open/close the fingers; with wait, the arm stands still meanwhile (as on the real robot).
        Closing on a graspable body attaches it to the hand, opening releases it."""
        changed = open_ != self.gripper_open
        self.gripper_open = open_
        closed = FINGER_CLOSED
        if changed and open_ and self.held:
            p.removeConstraint(self.held[1])
            self.held = None
        elif changed and not open_:
            self.held = self._attach()
            if self.held:
                closed = 0.006                    # fingers rest on the handle instead of passing through it
        p.setJointMotorControlArray(self.panda, FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[FINGER_OPEN if open_ else closed] * 2, forces=[20, 20])
        if changed and wait:
            self.wait(SIM_GRIPPER_TIME)

    def _attach(self, reach=0.03):
        """Fix the graspable body within `reach` of the grasp target to the hand, keeping its current pose."""
        hand = self.ee()
        for body in self.graspable:
            low, high = (np.array(corner) for corner in p.getAABB(body))
            if np.linalg.norm(hand - np.clip(hand, low, high)) < reach:
                hand_pos, hand_orn = p.getLinkState(self.panda, EE_LINK, computeForwardKinematics=True)[:2]
                body_pos, body_orn = p.getBasePositionAndOrientation(body)
                inv = p.invertTransform(hand_pos, hand_orn)
                rel_pos, rel_orn = p.multiplyTransforms(*inv, body_pos, body_orn)
                constraint = p.createConstraint(self.panda, EE_LINK, body, -1, p.JOINT_FIXED, [0, 0, 0],
                                                parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
                                                parentFrameOrientation=rel_orn)
                p.changeConstraint(constraint, maxForce=200)
                return body, constraint
        return None

    def step(self):
        if not p.isConnected():
            raise SystemExit("[INFO] Simulation window closed.")
        p.stepSimulation()
        self.time += SIM_DT
        if self.trail and round(self.time / SIM_DT) % 12 == 0:       # trail at 20 Hz
            ee = self.ee()
            if self.last_ee is not None:
                p.addUserDebugLine(self.last_ee, ee, self.color, lineWidth=3)
            self.last_ee = ee
        if self.frames is not None and self.time >= self.next_frame:  # video frames at VIDEO_FPS (video time)
            self.frames.append(self.render())
            self.next_frame += self.speed / VIDEO_FPS
        if self.gui:                                                  # real time (scaled by speed)
            if self.wall_start is None:                               # clock starts with the first step
                self.wall_start = time.monotonic() - self.time / self.speed
            time.sleep(max(0.0, self.wall_start + self.time / self.speed - time.monotonic()))

    def wait(self, seconds):
        for _ in range(round(seconds / SIM_DT)):
            self.step()

    def record(self):
        """Collect a video frame every 1/VIDEO_FPS s of video time from now on (see render)."""
        self.frames, self.next_frame = [], self.time

    def render(self):
        from PIL import Image
        width, height = VIDEO_SIZE
        c = self.camera
        view = p.computeViewMatrixFromYawPitchRoll(c["cameraTargetPosition"], c["cameraDistance"], c["cameraYaw"],
                                                   c["cameraPitch"], 0, 2)
        projection = p.computeProjectionMatrixFOV(50, width / height, 0.05, 10)
        rgba = p.getCameraImage(width, height, view, projection, shadow=1, lightDirection=[0.5, -1.0, 2.0],
                                renderer=p.ER_TINY_RENDERER)[2]
        return Image.fromarray(np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3])

    def show(self, text, color):
        print(f"[SIM] {text}")
        self.trail, self.color = True, color
        options = {} if self.label is None else {"replaceItemUniqueId": self.label}
        self.label = p.addUserDebugText(text, [0.3, 0.0, 1.05], textColorRGB=color, textSize=2.0, **options)

    def mark_waypoints(self, waypoints):
        """Small numbered spheres at the end-effector position of every waypoint."""
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)       # hide the robot jumping through them
        sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[*MARKER_COLOR, 1])
        q, gripper_open = self.q(), self.gripper_open
        for i, wp in enumerate(waypoints, 1):
            self.reset(wp["joint_positions"], gripper_open)
            position = self.ee()
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=position)
            p.addUserDebugText(str(i), position + [0, 0, 0.03], textColorRGB=MARKER_COLOR, textSize=1.2)
        self.reset(q, gripper_open)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def hold(self):
        """Keep the final pose on screen until the window is closed (or Ctrl+C)."""
        if not self.gui:
            return
        print("[INFO] Close the simulation window (or press Ctrl+C) to exit.")
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            pass


# ---------------- scenes ----------------
def pnp_scene(sim, waypoints):
    """Red mug whose handle (facing the robot) is where the gripper first closes, on a table at that
    height, and a toaster between the mug and the goal."""
    s = PNP_CUP_SCALE
    handle = np.array([-0.073, 0.0, 0.05]) * s     # grasp point on the handle, mug turned so the handle faces -x
    grasp = next(w for w in waypoints if not w["gripper_open"])
    cup = np.array(sim.hand_pose(grasp["joint_positions"])[0]) - handle
    sim.add_table(cup[2])
    yaw = p.getQuaternionFromEuler([0, 0, np.pi / 2])            # mug.urdf handle points to +y -> -x
    mug = p.loadURDF(os.path.join(DATA, "objects/mug.urdf"), basePosition=cup, baseOrientation=yaw, globalScaling=s)
    p.changeVisualShape(mug, -1, rgbaColor=[0.85, 0.1, 0.1, 1])
    k = PNP_TOASTER["scale"]
    p.loadURDF(str(ROOT / "assets" / "toaster.urdf"), globalScaling=k, baseOrientation=yaw,
               basePosition=[PNP_TOASTER["x"], PNP_TOASTER["y"], cup[2] + 0.075 * k])
    sim.no_collision(mug)                          # the hand holds it through a constraint (set_gripper)
    sim.graspable.append(mug)


SCENES = {"pnp": pnp_scene}


def make_simulation(base_name, waypoints, gui=True, speed=1.0):
    """Scene for the task's base name (pnp_spill -> "pnp"); robot and table only for other tasks."""
    sim = Simulation(gui, speed, SCENE_CAMERAS.get(base_name, SIM_CAMERA))
    if base_name in SCENES and not all(w["gripper_open"] for w in waypoints):
        SCENES[base_name](sim, waypoints)
        sim.wait(0.3)                              # let the objects settle on the table
        sim.time = 0.0
    else:
        sim.add_table()
    return sim


# ---------------- tasks and corrections ----------------
def plan_task(waypoints):
    """[(JointSpline, gripper state after it, last waypoint index)] for the task driven from home."""
    plan, q = [], HOME
    for points, gripper_open, last in task_segments(waypoints, True):
        plan.append((JointSpline([q, *points], VMAX, AMAX), gripper_open, last))
        q = points[-1]
    return plan


def play_task(sim, plan, stop=None):
    """Drive the plan; stop=(segment, time) ends it there (where a correction started)."""
    for k, (spline, gripper_open, _) in enumerate(plan):
        end = stop[1] if stop and stop[0] == k else spline.duration
        for t in np.arange(0.0, end, SIM_DT):
            sim.command(*spline.sample(t))
            sim.step()
        if stop and stop[0] == k:
            return
        sim.command(spline.sample(spline.duration)[0])
        sim.set_gripper(gripper_open)


def save_video(frames, path):
    """Animated WebP (Pillow only); plays in browsers and in the experiment GUI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / VIDEO_FPS), loop=0,
                   quality=80, method=4)


def newest_correction(task_name):
    paths = sorted((CORRECTION_DIR / task_name).glob("correction_*.json"))
    return paths[-1] if paths else None


def load_correction(path):
    data = json.loads(path.read_text())
    if not data.get("trajectory"):
        raise ValueError(f"{path} contains no trajectory")
    return data


def find_correction_start(plan, q0, gripper0):
    """(segment, time, distance [rad]) of the planned task moment closest to the correction's first
    configuration, among moments with the same gripper state. Correction files do not store where the
    task was interrupted, so this is reconstructed from the configuration where the correction began."""
    best = (np.inf, 0, 0.0)
    for same_gripper_only in (True, False):
        gripper_open = True
        for k, (spline, gripper_after, _) in enumerate(plan):
            if gripper_open == gripper0 or not same_gripper_only:
                for t in np.linspace(0.0, spline.duration, max(2, int(spline.duration * 50))):
                    distance = np.max(np.abs(spline.sample(t)[0] - q0))
                    if distance < best[0]:
                        best = (distance, k, t)
            gripper_open = gripper_after
        if np.isfinite(best[0]):
            break
    return best[1], best[2], best[0]


def play_correction(sim, correction):
    """Replay the dense correction. Samples are spaced 1/state_rate_hz on the NUC (the stored t are host
    receive times, which bunch up), so the nominal spacing is used."""
    q = np.array([sample["q"] for sample in correction["trajectory"]])
    gripper = [sample["gripper_open"] for sample in correction["trajectory"]]
    t = np.arange(len(q)) / correction["state_rate_hz"]
    qd = np.gradient(q, t, axis=0) if len(q) > 1 else np.zeros_like(q)
    start = JointSpline([sim.q(), q[0]], VMAX, AMAX)           # the robot stood here when the correction began
    for s in np.arange(0.0, start.duration, SIM_DT):
        sim.command(*start.sample(s))
        sim.step()
    for s in np.arange(0.0, t[-1] + SIM_DT, SIM_DT):
        i = min(int(s * correction["state_rate_hz"]), len(q) - 1)
        sim.command([np.interp(s, t, q[:, j]) for j in range(7)], qd[i])
        sim.set_gripper(gripper[i], wait=False)
        sim.step()
