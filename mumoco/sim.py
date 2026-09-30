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

from .config import AMAX, CORRECTION_DIR, SIM_ARM_FORCES, SIM_CAMERA, SIM_DT, SIM_GRIPPER_TIME, VMAX
from .franka import Franka
from .trajectory_utils import JointSpline, task_segments

HOME = Franka().home
EE_LINK = 11                        # panda_grasptarget in pybullet_data's panda.urdf
FINGERS = [9, 10]
FINGER_OPEN, FINGER_CLOSED = 0.04, 0.0
MARKER_COLOR = [102 / 255, 102 / 255, 102 / 255]


class Simulation:
    """Panda on a table. show() puts a large status label above the robot; the end-effector trail is
    drawn in the color of the current status."""

    def __init__(self, gui=True, speed=1.0):
        self.gui, self.speed = gui, speed
        p.connect(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(**SIM_CAMERA)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_DT)
        root = pybullet_data.getDataPath()
        scenery = [p.loadURDF(os.path.join(root, "plane.urdf"), basePosition=[0, 0, -0.625]),
                   p.loadURDF(os.path.join(root, "table/table.urdf"), basePosition=[0.5, 0, -0.625])]
        self.panda = p.loadURDF(os.path.join(root, "franka_panda/panda.urdf"), useFixedBase=True)
        for body in scenery:        # the recorded motions are real; the scene is only for orientation
            for link in range(-1, p.getNumJoints(self.panda)):
                for other in range(-1, p.getNumJoints(body)):
                    p.setCollisionFilterPair(self.panda, body, link, other, 0)
        self.label, self.color, self.last_ee = None, MARKER_COLOR, None
        self.time, self.wall_start, self.gripper_open = 0.0, time.monotonic(), True
        self.reset(HOME)

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
        """Open/close the fingers; with wait, the arm stands still meanwhile (as on the real robot)."""
        changed = open_ != self.gripper_open
        self.gripper_open = open_
        p.setJointMotorControlArray(self.panda, FINGERS, p.POSITION_CONTROL,
                                    targetPositions=[FINGER_OPEN if open_ else FINGER_CLOSED] * 2, forces=[20, 20])
        if changed and wait:
            self.wait(SIM_GRIPPER_TIME)

    def step(self):
        if not p.isConnected():
            raise SystemExit("[INFO] Simulation window closed.")
        p.stepSimulation()
        self.time += SIM_DT
        if round(self.time / SIM_DT) % 12 == 0:                      # trail at 20 Hz
            ee = self.ee()
            if self.last_ee is not None:
                p.addUserDebugLine(self.last_ee, ee, self.color, lineWidth=3)
            self.last_ee = ee
        if self.gui:                                                  # real time (scaled by speed)
            time.sleep(max(0.0, self.wall_start + self.time / self.speed - time.monotonic()))

    def wait(self, seconds):
        for _ in range(round(seconds / SIM_DT)):
            self.step()

    def show(self, text, color):
        print(f"[SIM] {text}")
        self.color = color
        options = {} if self.label is None else {"replaceItemUniqueId": self.label}
        self.label = p.addUserDebugText(text, [0.3, 0.0, 1.05], textColorRGB=color, textSize=2.0, **options)

    def mark_waypoints(self, waypoints):
        """Small numbered spheres at the end-effector position of every waypoint."""
        sphere = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[*MARKER_COLOR, 1])
        q, gripper_open = self.q(), self.gripper_open
        for i, wp in enumerate(waypoints, 1):
            self.reset(wp["joint_positions"], gripper_open)
            position = self.ee()
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=sphere, basePosition=position)
            p.addUserDebugText(str(i), position + [0, 0, 0.03], textColorRGB=MARKER_COLOR, textSize=1.2)
        self.reset(q, gripper_open)

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
