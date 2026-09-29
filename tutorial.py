import sys, time, select, signal, numpy as np                        # timing, arrays, Ctrl+C handling
from franka import Franka                                             # fixed franka.py in the same folder

HZ = 20                                                               # command rate = rate the NUC (cjw) reads commands (policy_frequency)
LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])  # Panda joint limits [rad]
UPPER = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
MARGIN = 0.05                                                         # stay this far [rad] inside the limits

# ---------------- waypoints (edit here) ----------------
# ('joint', [q1..q7])               joint configuration [rad]
# ('xyz',   [x, y, z])              TCP position [m] in the base frame, orientation kept from the previous waypoint
# ('pose',  [x, y, z, r, p, yaw])   TCP position + roll/pitch/yaw [rad], same convention as state['x']
# ('grip',  'o' or 'c')             open / close the gripper
# TCP = flange + 0.107 + 0.20 m (franka.joint2pose). At home the TCP is ~[0.40, -0.03, 0.33], pointing down.
WAYPOINTS = [
    ('grip', 'o'),
    ('xyz',  [0.45,  0.10, 0.30]),                                    # above the pick point
    ('xyz',  [0.45,  0.10, 0.22]),                                    # go down
    ('grip', 'c'),
    ('xyz',  [0.45,  0.10, 0.30]),                                    # lift
    ('xyz',  [0.45, -0.15, 0.30]),                                    # above the place point
    ('xyz',  [0.45, -0.15, 0.22]),                                    # go down
    ('grip', 'o'),
    ('xyz',  [0.45, -0.15, 0.30]),                                    # lift
    ('joint', Franka().home),                                         # joint targets work too
]


def rpy2rot(r, p, y):                                                 # inverse of the roll/pitch/yaw in franka.listen2robot (R = Rz Ry Rx)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])

def rot_error(R_goal, R):                                             # axis * angle that rotates R onto R_goal (base frame)
    E = R_goal @ R.T
    v = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    theta = np.arccos(np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))
    return v * theta / np.sin(theta) if np.sin(theta) > 1e-4 else v

def ik(p_goal, R_goal, q_seed, iters=500):                            # Cartesian goal -> joint goal (damped least squares on joint2pose)
    q = np.array(q_seed, dtype=np.float64)
    for _ in range(iters):
        p, R = Franka.joint2pose(q)
        e = np.concatenate((p_goal - p, rot_error(R_goal, R)))       # 6D error [m, rad]
        if np.linalg.norm(e[:3]) < 1e-4 and np.linalg.norm(e[3:]) < 1e-3:
            return q
        J = np.zeros((6, 7))                                          # numerical Jacobian of the same FK (same TCP as the goal)
        for i in range(7):
            dq = np.zeros(7); dq[i] = 1e-6
            p2, R2 = Franka.joint2pose(q + dq)
            J[:3, i] = (p2 - p) / 1e-6; J[3:, i] = rot_error(R2, R) / 1e-6
        step = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), e)
        step *= min(1.0, 0.2 / np.linalg.norm(step))                  # small steps -> stays near the seed configuration
        q = np.clip(q + step, LOWER + MARGIN, UPPER - MARGIN)
    raise ValueError(f'IK failed for p={np.round(p_goal, 3)} (residual {np.linalg.norm(e[:3]):.4f} m)')

def make_plan(waypoints, q_start):                                    # every waypoint -> ('joint', q) or ('grip', c), solved before moving
    plan, q, R = [], np.array(q_start, dtype=np.float64), Franka.joint2pose(q_start)[1]
    for kind, val in waypoints:
        if kind == 'grip':
            plan.append(('grip', val)); continue
        if kind == 'joint':
            q = np.asarray(val, dtype=np.float64)
            if np.any(q < LOWER) or np.any(q > UPPER): raise ValueError(f'joint target outside limits: {q}')
        else:                                                         # 'xyz' or 'pose'
            if kind == 'pose': R = rpy2rot(*val[3:6])
            q = ik(np.asarray(val[:3], dtype=np.float64), R, q)       # seed = previous target -> smooth sequence
        R = Franka.joint2pose(q)[1]
        plan.append(('joint', q))
    return plan


plan = make_plan(WAYPOINTS, Franka().home)                            # plan from home (the robot goes home first)
for (kind, val), (_, target) in zip(plan, WAYPOINTS):
    print(kind, np.round(val, 3) if kind == 'joint' else val, ' <-', target if kind == 'grip' else np.round(target, 3))
if 'dry' in sys.argv[1:]: sys.exit()                                  # python3 tutorial.py dry : check the plan without the robot

robot = Franka()                                                      # home pose + bind address (172.16.0.3)
conn, grip = robot.connect_robot_and_gripper(8080, 8081); print('connected')  # open both ports, wait for NUC controllers (any start order)
s = robot.readState(conn); print('q:', s['q'].round(3), 'x:', s['x'].round(3))  # 7 joint angles, TCP [xyz, euler]

class Correction(Exception): pass                                     # raised inside a motion when Ctrl+C was pressed
mode = {'correction': False, 'requested': False}

def on_ctrl_c(sig, frame):                                            # 1st Ctrl+C: request correction mode, 2nd Ctrl+C: quit
    if mode['requested']: raise KeyboardInterrupt
    mode['requested'] = True                                          # only a flag -> never cuts a send2robot message in half

def move_joint(q_goal, vmax=0.3, K=1.5, tol=0.01, timeout=20.0):     # P-control in joint space: qdot = K (q_goal - q), all joints arrive together
    start = time.monotonic()
    while True:
        if mode['requested']: raise Correction
        t0 = time.monotonic(); q = robot.readState(conn)['q']
        err = q_goal - q
        if np.max(np.abs(err)) < tol: break
        if t0 - start > timeout: raise TimeoutError(f'joint error still {np.round(err, 3)}')
        qdot = K * err
        qdot *= min(1.0, vmax / np.max(np.abs(qdot)))                 # scale the whole vector -> straight line in joint space
        robot.send2robot(conn, qdot); time.sleep(max(0.0, 1 / HZ - (time.monotonic() - t0)))  # send at 20 Hz
    robot.send2robot(conn, np.zeros(7)); time.sleep(0.3)              # stop and let the robot settle
    s = robot.readState(conn); print('q:', s['q'].round(3), 'x:', s['x'][:3].round(3))

def gripper(cmd, wait=2.0):                                           # 'o' open (move to 7.5 cm) / 'c' close (grasp, 5 N)
    robot.send2gripper(grip, cmd); time.sleep(wait)                   # the NUC gives no "done" signal -> just wait

def correction():                                                     # hand-guiding; the NUC keeps streaming the state
    robot.send2mode(conn, 'c'); mode['correction'] = True             # NUC: velocity command -> 0 (from here on: no send2robot!)
    print('\n[correction] press external activation and guide the robot.'
          '\n[correction] release it, then press Enter to resume (Ctrl+C again: quit)')
    while not select.select([sys.stdin], [], [], 0.05)[0]:            # TODO: replace with the "end correction" UI
        s = robot.readState(conn)                                     # latest state while guiding (record corrections here)
    sys.stdin.readline()
    robot.send2mode(conn, 'v'); time.sleep(0.5)                       # NUC: back to velocity control
    mode['correction'] = mode['requested'] = False
    print('[correction] done, resuming')

input('Enter to start (robot goes home first) ')                      # NUC (cjw) starts in velocity control mode, no "v" needed
signal.signal(signal.SIGINT, on_ctrl_c)
steps = [('joint', robot.home)] + plan + [('joint', robot.home)]      # home -> waypoints -> home
try:
    i = 0
    while i < len(steps):
        kind, val = steps[i]
        print('->', kind, np.round(val, 3) if kind == 'joint' else val)
        try:
            move_joint(val) if kind == 'joint' else gripper(val)
            i += 1
        except Correction:
            correction()                                              # then retry the same waypoint from wherever the robot was guided to
finally:
    if not mode['correction']:                                        # in correction mode the NUC only accepts 'v'/'c'
        robot.send2robot(conn, np.zeros(7))                           # always send a stop command (errors, Ctrl+C)
