"""
Code taken from:https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py
@article{black2410pi0,
  title={$pi$0: A vision-language-action flow model for general robot control. CoRR, abs/2410.24164, 2024. doi: 10.48550},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint ARXIV.2410.24164}
}
"""
import os
import h5py
import shutil
import natsort
import numpy as np
from pathlib import Path
from termcolor import colored
from omegaconf import DictConfig 

from utils import preprocess_video
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

task_name = "SSA"
data_path = "/projects/collab/Yinlong/semantic_shared_autonomy/demos"
num_demos = 90
def main() -> None:
    output_path = HF_LEROBOT_HOME / f"{task_name}_{num_demos}"
    # if output_path.exists():
    #     shutil.rmtree(output_path)
    if output_path.exists():
        shutil.rmtree(f"/home/daiyinlong/.cache/huggingface/lerobot/{task_name}_{num_demos}")

    # os.makedirs(output_path, exist_ok=True)

    dataset = LeRobotDataset.create(
        repo_id = f"{task_name}_{num_demos}",
        robot_type = "panda",
        # Match the DROID collection/control frequency.
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            # Kept for compatibility with the official DROID LeRobot schema.
            # pi05_droid uses the first exterior camera and masks this view.
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            }
        },
        image_writer_threads = 10,
        image_writer_processes = 5
    )


    files = natsort.humansorted(
            [f for f in os.listdir(data_path) if f.endswith(".h5") or f.endswith(".hdf5")]
        )[:num_demos]

    print(files)

    def print_hdf5_keys(f):
        def visitor(name, obj):
            print(name)
        f.visititems(visitor)

    for file in files:
        print(colored("[Processing] ", "green") + f"Processing file: {file}")

        file_path = os.path.join(data_path, file)
        with h5py.File(file_path, "r") as f:
            print("=== HDF5 Keys ===")
            print_hdf5_keys(f)
            print("=================")
            data = {}
            data["joint_vel"] = f["actions/joint_vel"][:]        
            data["joint_pos"] = f["obs/joint_state"][:]    
            data["left_rgb"]  = f["obs/side_img_rgb"][:]
            data["gripper_rgb"] = f["obs/gripper_img_rgb"][:]
            data["gripper_state"] = f["obs/gripper_state"][:]
            data["gripper_action"] = f["actions/gripper_action"][:]
            data["task_description"] = f.attrs["task_description"]
            data["fps"] = f.attrs.get("fps")
        assert data["joint_vel"].shape[1] == 7, "Expected joint_vel to have shape (T, 7)"
        print(f"Loaded data with {len(data['left_rgb'])} frames.")
        print(f"Task description: {data['task_description']}")
        data['left_rgb'] = np.clip(
            preprocess_video(np.array(data['left_rgb']) / 255.0, specs = {'resize': (224, 224)}),
            0, 1
        )
        data['gripper_rgb'] = np.clip(
            preprocess_video(np.array(data['gripper_rgb']) / 255.0, specs = {'resize': (224, 224)}),
            0, 1
        )

        for step in range(len(data["left_rgb"])):
            dataset.add_frame(
                {
                    "exterior_image_1_left": data["left_rgb"][step],
                    # pi05-DROID does not use this view; retain the schema with a black placeholder.
                    "exterior_image_2_left": np.zeros_like(data["left_rgb"][step]),
                    "wrist_image_left": data["gripper_rgb"][step],
                    "joint_position": np.array(data["joint_pos"][step], dtype="float32"),
                    "gripper_position": np.asarray(data["gripper_state"][step], dtype="float32").reshape(1),
                    "actions": np.append(np.array(data["joint_vel"][step] , dtype = "float32"), data["gripper_action"][step]),
                    "task": data["task_description"]
                }
            )
        dataset.save_episode()


if __name__ == "__main__":
    main()
