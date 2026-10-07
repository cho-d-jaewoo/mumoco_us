from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.robots.dynamixel import DynamixelRobot
from gello.agents.gello_agent import GelloAgent
from gello.env import RobotEnv
from gello.zmq_core.robot_node import ZMQClientRobot

@dataclass
class Args:
     hostname: str = "127.0.0.1"
     gello_port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAA0ARK-if00-port0"
     robot_port: int = 6001
     start_joints: Optional[Tuple[float, ...]] = None

def main(args):
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    agent = GelloAgent(port=args.gello_port, start_joints=args.start_joints)
    print(args.start_joints)

    max_joint_delta = 0.3
    locked = [False, False, False, False, False, False, False]

    while(not all(locked)):
        start_pos = agent.act(env.get_obs())
        for pos_id in range(len(start_pos)):
            if 6.0 < start_pos[pos_id] < 6.6:
                start_pos[pos_id] = start_pos[pos_id] - (2 * np.pi)
            if -6.6 < start_pos[pos_id] < -6.0:
                start_pos[pos_id] = start_pos[pos_id] + (2 * np.pi)

        obs = env.get_obs()
        joints = obs["joint_positions"]
        abs_deltas = np.abs(start_pos - joints)

        for id in range(len(locked)):
            if locked[id] == False:
                if abs_deltas[id] < max_joint_delta:
                    agent._robot.set_individual_torque(id + 1)
                    locked[id] = True
                    print(f"Joint {id} locked!")

    if all(locked):
        print("All joints locked!")
        exit()

                    




if __name__ == "__main__":
    main(tyro.cli(Args))