import os
import time
import hydra
from typing import Union
from termcolor import colored
from omegaconf import DictConfig, OmegaConf

from codata.utils import save_to_hdf5
from codata.io_devices.keyboard import Keyboard
from codata.io_devices.gello_teleop import Gello
from codata.cameras.cameras_subscriber import CamerasSubscriber
from codata.cameras.orbbec_subscriber import OrbbecSubscriber


def teleop_demo(
                gello: Gello,
                interface: Keyboard, 
                cameras_sub: list[Union[CamerasSubscriber, OrbbecSubscriber]],
                record_freq: float,):
    record_period = 1. / record_freq

    # Data to be saved
    observations = {'joint_state': [],
                    'ee_state': [],
                    'gripper_state': [],
                    'timestep': []
                    }
    
    for sub in cameras_sub:
        for topic in sub.topics:
            observations[topic] = []
    print(colored(f"[Recording] ,","green") + f" the following topics will be recorded: {observations.keys()}")

    actions = {'joint_vel': [],
            'ee_vel': [],
            'gripper_action': []
            }
    
    record = False
    run = True
    last_time = None
    gello._go_home()
    print("Press A to start recording...")
    while run:
        start_step = time.time_ns()
    
        if interface.a_pressed and not record:
            print(f"Starting Lock and Check")
            gello.lock_and_check()
            gello.run()
            time.sleep(2)
            record = True
            print("[*] Recording Started")
            time.sleep(0.1)
            start_time = time.time()
        if interface.b_pressed and record:
            record = False
            print(f"This demo has: {len(observations['joint_state'])} data points")
            print("[*] Press A to continue recording or START to save the recorded demo")
            gello.done = True
            time.sleep(1.0)
        if interface.s_pressed:
            run = False

        # Get current camera frames
        frames = cameras_sub[0].get_last_frames()
        if len(cameras_sub) > 1 and isinstance(cameras_sub[1], OrbbecSubscriber):
            pass # Get current Orbbec frames if OrbbecSubscriber is used

        if record:
            obs = gello.obs

            observations['joint_state'].append(obs['joint_positions'][:7])
            observations['ee_state'].append(obs['ee_pos_quat'])
            observations['gripper_state'].append(1 if len(actions['gripper_action']) == 0 else actions['gripper_action'][-1])
            observations['timestep'].append(obs['timestep'])
            for key, img in frames.items():
                observations[key].append(img)

            actions['joint_vel'].append(obs['joint_velocities'][:7])
            actions['ee_vel'].append(obs['ee_velocities'])
            actions['gripper_action'].append(obs['gripper_binary_state'])
        
        if last_time == None:
            last_time = time.time_ns()

        curr_time = time.time_ns()    
        remaining_time = record_period - ((curr_time - last_time) / (10**9))
        if (0.0001 < remaining_time < 0.2):
            time.sleep(remaining_time)
        last_time = time.time_ns()
        end_step = time.time_ns()
        # print(f"[*] Time profile: {(end_step - start_step) / 10**9}")
    
    if gello.gello_thread.is_alive():
        gello.done = True
    
    return observations, actions

@hydra.main(version_base=None, config_path='src/config', config_name='get_demo')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    os.makedirs(cfg.save_dir, exist_ok=True)
    demo_list = os.listdir(cfg.save_dir)
    save_number = len(demo_list)
    save_path = f'{cfg.save_dir}/demo_{save_number}.hdf5'
    print(f"[*] Saving first demo to: {save_path}")

    # Initialize camera subscriber
    cameras_sub = CamerasSubscriber(topics=cfg.camera_subscriber.topics,
                                    server_addr=cfg.camera_subscriber.server_addr,
                                    port=cfg.camera_subscriber.port)
    cameras_sub.start_thread()

    orbbec_sub = None
    if "orbbec_subscriber" in cfg:
        pass # Initialize OrbbecSubscriber if needed
    cameras_sub = [cameras_sub, orbbec_sub] if orbbec_sub is not None else [cameras_sub]

    # Initalize i/o device
    # interface = Joystick(step_size_l=cfg.linear_speed)
    interface = Keyboard()
    gello = Gello(home_pos=cfg.task.start_position)

    try:
        saved_demo_count = 0
        while saved_demo_count < cfg.num_demos:
            # Collect demonstration
            observations, actions = teleop_demo( 
                                                gello,
                                                interface, 
                                                cameras_sub, 
                                                cfg.record_freq,)
            print(f"This demo has: {len(observations['joint_state'])} data points")
            print(f"Press X on the keyboard to save and Y to discard")

            if len(observations) == 0:
                print("no datapoints to save")
                continue

            while True:
                if interface.x_pressed:
                    labels = []
                    # finish_input = False
                    # while not finish_input:
                    #     for i, label in enumerate(cfg.task.labels):
                    #         print(f"{i}: {label}")
                    #     label = input(colored("[Label] ", "yellow") + "Enter the index of a label from the above list (or type 'done' to finish):")
                    #     label = label[-4:] if 'done' in label else label[-1] 
                    #     try:
                    #         assert label.lower() == 'done' or label in [str(x) for x in range(len(cfg.task.labels))]
                    #     except: 
                    #         print("Invalid label, type again:")
                    #         continue
                    #     if label.lower() == 'done':
                    #         finish_input = True
                    #     else:
                    #         labels.append(int(label))
                    save_path = f'{cfg.save_dir}/demo_{save_number}.hdf5'
                    # if cfg.task.task_name == 'pick_and_place':
                    #     save_dir = f'{cfg.save_dir}/{cfg.task.labels[labels[0]]}'
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     demo_list = os.listdir(cfg.save_dir)
                    #     save_number = len(demo_list) + 1
                    #     save_path = f'{save_dir}/demo_{save_number}.hdf5'

                    attrs = {'fps': cfg.record_freq,
                            'record_freq': cfg.record_freq,
                            'task_description': cfg.task_description,
                            'labels': labels
                            }
                    save_to_hdf5(save_path, observations, actions, attrs)
                    print(f"[*] Saving demo to: {save_path}")
                    
                    save_number += 1
                    saved_demo_count +=1
                    break
                elif interface.y_pressed:
                    break
            
    except KeyboardInterrupt:
        print("[*] The script was terminating sending robot back home")
        if 'gello' in locals():
            if hasattr(gello, 'gello_thread') and gello.gello_thread.is_alive():
                gello.done = True
                gello.gello_thread.join()
            gello._go_home()
    finally:
        if 'gello' in locals():
            if hasattr(gello, 'gello_thread') and gello.gello_thread.is_alive():
                gello.done = True
                gello.gello_thread.join()
            if hasattr(gello, 'close'):
                gello.close()
        if 'cameras_sub' in locals():
            for sub in cameras_sub:
                if sub is not None:
                    sub.close_subscriber()
        if 'interface' in locals() and hasattr(interface, 'close'):
            interface.close()
        print(f"done")


if __name__== '__main__':
    main()
