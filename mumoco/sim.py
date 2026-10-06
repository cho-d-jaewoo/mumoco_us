"""PyBullet replay of recorded tasks and corrections (scene and Panda as in panda-tutorial / lfc_test).

The task is driven exactly as Robot.execute_task drives the real robot: the same task_segments and
JointSpline, starting at home with the gripper open. The robot follows the reference with position control.
"""

import json
import os
import tempfile
import time

import numpy as np
import pybullet as p
import pybullet_data

from .config import (AMAX, BIT_PLATE, CORRECTION_DIR, CORRECTION_FOLDERS, PNP_CUP_SCALE, PNP_TOASTER, ROOT,
                     SCENE_CAMERAS, SIM_ARM_FORCES, SIM_CAMERA, SIM_DT, SIM_GRIPPER_TIME, VIDEO_FPS, VIDEO_SIZE, VMAX, WTP_PLATE)
from .franka import Franka
from .trajectory_utils import JointSpline, task_segments
from .utils import find_task_path, load_task

HOME = Franka().home
EE_LINK = 11                        # panda_grasptarget in pybullet_data's panda.urdf
FINGERS = [9, 10]
FINGER_OPEN, FINGER_CLOSED = 0.04, 0.0
DATA = pybullet_data.getDataPath()
TABLE_HEIGHT = 0.626                # table.urdf: top surface above its base


class Simulation:
    """Panda next to a table; only the robot, the scene and its objects are shown."""

    def __init__(self, gui=True, speed=1.0, camera=SIM_CAMERA, deformable=False):
        self.gui, self.speed, self.camera = gui, speed, camera
        p.connect(p.GUI if gui else p.DIRECT)
        if deformable:                              # soft bodies (cloth); only scenes that need them
            p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(**camera)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIM_DT)
        self.panda = p.loadURDF(os.path.join(DATA, "franka_panda/panda.urdf"), useFixedBase=True)
        self.graspable, self.held = [], None       # bodies the gripper can pick up / (body, [constraints])
        self.cloths = {}                           # soft cloth the gripper can pick up -> prop under it (or None)
        self.time, self.wall_start, self.gripper_open = 0.0, None, True
        self.frames, self.next_frame = None, 0.0
        self.size = VIDEO_SIZE                     # [px] rendered frames (width, height)
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
            for constraint in self.held[1]:
                p.removeConstraint(constraint)
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
                return body, [constraint]
        for cloth, prop in self.cloths.items():    # soft cloth: hold the nodes near the fingertips
            nodes = np.array(p.getMeshData(cloth, -1, flags=p.MESH_DATA_SIMULATION_MESH)[1])
            near = np.flatnonzero(np.linalg.norm(nodes - hand, axis=1) < CLOTH["reach"])
            if len(near):
                if prop is not None:               # the prop is no longer needed (and would show once uncovered)
                    p.removeBody(prop)
                    self.cloths[cloth] = None
                return cloth, [p.createSoftBodyAnchor(cloth, int(i), self.panda, EE_LINK) for i in near]
        return None

    def step(self):
        if not p.isConnected():
            raise SystemExit("[INFO] Simulation window closed.")
        p.stepSimulation()
        self.time += SIM_DT
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
        width, height = self.size
        c = self.camera
        view = p.computeViewMatrixFromYawPitchRoll(c["cameraTargetPosition"], c["cameraDistance"], c["cameraYaw"],
                                                   c["cameraPitch"], 0, 2)
        projection = p.computeProjectionMatrixFOV(50, width / height, 0.05, 10)
        rgba = p.getCameraImage(width, height, view, projection, shadow=1, lightDirection=[0.5, -1.0, 2.0],
                                renderer=p.ER_TINY_RENDERER)[2]
        return Image.fromarray(np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3])

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
def grasp_and_release(waypoints):
    """Joint positions where the gripper first closes and where it opens again after that."""
    k = next(i for i, w in enumerate(waypoints) if not w["gripper_open"])
    release = next((w for w in waypoints[k:] if w["gripper_open"]), waypoints[-1])
    return waypoints[k]["joint_positions"], release["joint_positions"]


def pnp_scene(sim, task, answer):
    """Red mug whose handle (facing the robot) is where the task itself grasps, on a table at that height,
    and a toaster between the mug and the goal. (Placed from each task, not from the answer: the toaster
    position is tuned to the cup heights of the recorded tasks.)"""
    s = PNP_CUP_SCALE
    handle = np.array([-0.073, 0.0, 0.05]) * s     # grasp point on the handle, mug turned so the handle faces -x
    cup = np.array(sim.hand_pose(grasp_and_release(task)[0])[0]) - handle
    sim.add_table(cup[2])
    yaw = p.getQuaternionFromEuler([0, 0, np.pi / 2])            # mug.urdf handle points to +y -> -x
    mug = p.loadURDF(os.path.join(DATA, "objects/mug.urdf"), basePosition=cup, baseOrientation=yaw, globalScaling=s)
    p.changeVisualShape(mug, -1, rgbaColor=[0.85, 0.1, 0.1, 1])
    k = PNP_TOASTER["scale"]
    p.loadURDF(str(ROOT / "assets" / "toaster.urdf"), globalScaling=k, baseOrientation=yaw,
               basePosition=[PNP_TOASTER["x"], PNP_TOASTER["y"], cup[2] + 0.075 * k])
    sim.no_collision(mug)                          # the hand holds it through a constraint (set_gripper)
    sim.graspable.append(mug)


BREAD = {"width": 0.11, "thickness": 0.015, "height": 0.125, "grasp": 0.105}   # [m]; grasped 2 cm below the top


def bread_slice(position):
    """Upright slice of bread (square bottom, round top), thin along y, bottom center at `position`."""
    w, t, h = BREAD["width"], BREAD["thickness"], BREAD["height"]
    r, box = w / 2, h - w / 2                       # round top radius, height of the square part
    along_y = p.getQuaternionFromEuler([np.pi / 2, 0, 0])        # cylinder axis along y (the thickness)
    shapes = dict(shapeTypes=[p.GEOM_BOX, p.GEOM_CYLINDER], halfExtents=[[r, t / 2, box / 2], [0, 0, 0]],
                  radii=[0, r], lengths=[0, t])
    frames = dict(positions=[[0, 0, box / 2], [0, 0, box]], orientations=[[0, 0, 0, 1], along_y])
    collision = p.createCollisionShapeArray(**shapes, collisionFramePositions=frames["positions"],
                                            collisionFrameOrientations=frames["orientations"])
    visual = p.createVisualShapeArray(**shapes, visualFramePositions=frames["positions"],
                                      visualFrameOrientations=frames["orientations"],
                                      rgbaColors=[[0.86, 0.66, 0.38, 1]] * 2)
    return p.createMultiBody(baseMass=0.03, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                             basePosition=position, baseInertialFramePosition=[0, 0, h / 2])


def bit_scene(sim, task, answer):
    """Bread in toaster: a slice of bread standing where the answer task grasps it, a toaster whose slot
    is where the answer task drops it, and a blue plate (BIT_PLATE)."""
    grasp_q, release_q = grasp_and_release(answer)
    grasp_pos, grasp_orn = sim.hand_pose(grasp_q)
    bottom = np.array(grasp_pos) - [0, 0, BREAD["grasp"]]
    sim.add_table(bottom[2])
    # where the answer task releases the bread: the bread keeps its pose relative to the hand
    inv = p.invertTransform(grasp_pos, grasp_orn)
    in_hand = p.multiplyTransforms(*inv, bottom.tolist(), [0, 0, 0, 1])
    drop = p.multiplyTransforms(*sim.hand_pose(release_q), *in_hand)[0]
    # toaster turned so its slots run along y; the slot nearer the robot (0.025 m off center) under the drop
    yaw = p.getQuaternionFromEuler([0, 0, np.pi / 2])
    p.loadURDF(str(ROOT / "assets" / "toaster.urdf"), baseOrientation=yaw,
               basePosition=[drop[0] + 0.025, drop[1], bottom[2] + 0.075])
    plate = p.createVisualShape(p.GEOM_CYLINDER, radius=BIT_PLATE["radius"], length=0.012,
                                rgbaColor=[42 / 255, 143 / 255, 189 / 255, 1])
    plate_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=BIT_PLATE["radius"], height=0.012)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=plate_collision, baseVisualShapeIndex=plate,
                      basePosition=[BIT_PLATE["x"], BIT_PLATE["y"], bottom[2] + 0.006])
    bread = bread_slice(bottom)
    sim.no_collision(bread)                        # the hand holds it through a constraint (set_gripper)
    sim.graspable.append(bread)


CLOTH = {"side": 0.26, "nodes": 17,                    # [m] square handkerchief, nodes per side
         "hump": 0.05, "radius": 0.03, "length": 0.12,     # laid over a round hump (cave) this high / wide / long
         "grip": 0.01, "reach": 0.025,                     # held 1 cm below the hump top; nodes this close are held
         "stiffness": 120, "bending": 0.02, "damping": 0.03, "mass": 0.05}  # floppy fabric (stiffness/mass per node
                                                                             # must stay low enough to be stable)


def cloth_mesh(side, n, hump, radius):
    """OBJ file of a square cloth draped over a round hump that runs along x through its center."""
    center = hump - radius                         # hump (cylinder) axis height
    xs = np.linspace(-side / 2, side / 2, n)

    def height(y):                                 # over the cylinder, then down its side to the table
        if abs(y) <= radius:
            return center + np.sqrt(radius**2 - y**2)
        return max(0.0, center - 2.0 * (abs(y) - radius))

    vertices = [(x, y, height(y) + 0.004) for y in xs for x in xs]
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i + 1                       # OBJ indices start at 1
            faces += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as obj:
        obj.writelines(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vertices)
        obj.writelines(f"f {a} {b} {c}\n" for a, b, c in faces)
    return obj.name


def wtp_scene(sim, task, answer):
    """Wipe the plate: a light, floppy handkerchief laid over a small round hump (like a cave) where the
    answer task grasps it, so the fingers can pinch its top, and a white plate where the answer task wipes.
    The hump is an invisible cylinder under the cloth, removed when the cloth is grasped."""
    grasp = np.array(sim.hand_pose(grasp_and_release(answer)[0])[0])
    table = grasp[2] + CLOTH["grip"] - CLOTH["hump"]
    sim.add_table(table)
    plate = dict(radius=WTP_PLATE["radius"])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, height=0.01, **plate),
                      baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=0.01, rgbaColor=[0.95, 0.95, 0.93, 1],
                                                               **plate),
                      basePosition=[WTP_PLATE["x"], WTP_PLATE["y"], table + 0.005])
    color = np.array([0.55, 0.72, 0.88])
    along_x = p.getQuaternionFromEuler([0, np.pi / 2, 0])
    hump = dict(radius=CLOTH["radius"])
    prop = p.createMultiBody(baseMass=0,
                             baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_CYLINDER, height=CLOTH["length"], **hump),
                             baseVisualShapeIndex=p.createVisualShape(p.GEOM_CYLINDER, length=CLOTH["length"],
                                                                      rgbaColor=[*(color * 0.6), 1], **hump),  # cave shade
                             basePosition=[grasp[0], grasp[1], table + CLOTH["hump"] - CLOTH["radius"]],
                             baseOrientation=along_x)
    sim.no_collision(prop)
    path = cloth_mesh(CLOTH["side"], CLOTH["nodes"], CLOTH["hump"], CLOTH["radius"])
    cloth = p.loadSoftBody(path, basePosition=[grasp[0], grasp[1], table], mass=CLOTH["mass"], useNeoHookean=0,
                           useBendingSprings=1, useMassSpring=1, springElasticStiffness=CLOTH["stiffness"],
                           springDampingStiffness=CLOTH["damping"], springBendingStiffness=CLOTH["bending"],
                           springDampingAllDirections=1, useSelfCollision=0, frictionCoeff=0.3, useFaceContact=1,
                           collisionMargin=0.003)
    os.remove(path)
    p.changeVisualShape(cloth, -1, rgbaColor=[*color, 1], flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
    sim.no_collision(cloth)                        # held through anchors; finger contact would blow the cloth up
    p.changeDynamics(cloth, -1, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)   # a resting cloth would freeze
    sim.cloths[cloth] = prop
    p.setPhysicsEngineParameter(numSubSteps=4)    # (after loading the cloth) its springs need smaller steps
    sim.wait(1.0)                                  # let the cloth drape over the hump before the video starts


SCENES = {"pnp": pnp_scene, "bit": bit_scene, "wtp": wtp_scene}
DEFORMABLE = {"wtp"}                               # scenes with soft bodies


def make_simulation(base_name, waypoints, gui=True, speed=1.0):
    """Scene for the task's base name (pnp_spill -> "pnp"); scenes get the task and the base's answer task.
    Robot and table only for other tasks."""
    sim = Simulation(gui, speed, SCENE_CAMERAS.get(base_name, SIM_CAMERA), deformable=base_name in DEFORMABLE)
    answer_path = find_task_path(f"{base_name}_answer")
    answer = load_task(answer_path)["waypoints"] if answer_path else waypoints
    if base_name in SCENES and not all(w["gripper_open"] for w in answer):
        SCENES[base_name](sim, waypoints, answer)
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
    """Newest physical or multimodal correction of the task that has a trajectory (a multimodal one may be speech only)."""
    paths = [path for folder in (CORRECTION_FOLDERS["physical"], CORRECTION_FOLDERS["multimodal"])
             for path in (CORRECTION_DIR / task_name / folder).glob("*.json")
             if json.loads(path.read_text()).get("trajectory")]
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


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


def play_correction(sim, correction, wait=0.0):
    """Replay the dense correction; wait: [s] from now until its first motion (the robot moves to its start
    and stands there meanwhile). The stored t are host receive times, which bunch up, and truncated messages
    were skipped, so the samples are spread evenly over the recorded time span (real duration, smooth)."""
    trajectory = correction["trajectory"]
    q = np.array([sample["q"] for sample in trajectory])
    gripper = [sample["gripper_open"] for sample in trajectory]
    t = np.linspace(0.0, trajectory[-1]["t"] - trajectory[0]["t"], len(q))
    rate = (len(q) - 1) / t[-1] if t[-1] > 0 else correction["state_rate_hz"]
    qd = np.gradient(q, t, axis=0) if len(q) > 1 and t[-1] > 0 else np.zeros_like(q)
    start = JointSpline([sim.q(), q[0]], VMAX, AMAX)           # the robot stood here when the correction began
    for s in np.arange(0.0, max(start.duration, wait), SIM_DT):
        sim.command(*start.sample(s))                         # (stays at the end after its duration)
        sim.step()
    for s in np.arange(0.0, t[-1] + SIM_DT, SIM_DT):
        i = min(int(s * rate), len(q) - 1)
        sim.command([np.interp(s, t, q[:, j]) for j in range(7)], qd[i])
        sim.set_gripper(gripper[i], wait=False)
        sim.step()
