"""
Convert demo/data/<task>/demo_<n>.hdf5 (demo/get_demo.py) to a LeRobot dataset in the DROID layout for pi0.5
fine-tuning. Run it in the openpi / LeRobot environment:

    python demo/preprocess_lerobot.py pnp              # -> $HF_LEROBOT_HOME/pnp_<number of demos>
    python demo/preprocess_lerobot.py pnp bit wtp      # one dataset with all three tasks (each frame keeps its prompt)
    python demo/preprocess_lerobot.py pnp --num-demos 50

Settings (data, cameras, image size) come from demo/config/get_demo.yaml, so the dataset matches what was recorded.
The LeRobot fps is the recorded record_freq.

Code taken from:https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py
@article{black2410pi0,
  title={$pi$0: A vision-language-action flow model for general robot control. CoRR, abs/2410.24164, 2024. doi: 10.48550},
  author={Black, Kevin and Brown, Noah and Driess, Danny and Esmail, Adnan and Equi, Michael and Finn, Chelsea and Fusai, Niccolo and Groom, Lachy and Hausman, Karol and Ichter, Brian and others},
  journal={arXiv preprint ARXIV.2410.24164}
}
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path(__file__).resolve().parent.parent                    # repository root
CONFIG = Path(__file__).parent / "config" / "get_demo.yaml"


def demo_files(cfg, tasks, num_demos):
    """demo_<n>.hdf5 of the tasks in demo order (the first num_demos of each task if given)."""
    files = []
    for task in tasks:
        paths = sorted((ROOT / cfg["save_dir"] / task).glob("demo_*.hdf5"), key=lambda p: int(p.stem.split("_")[-1]))
        files += paths[:num_demos] if num_demos else paths
    return files


def resize(images, size):
    """(T,H,W,3) uint8 -> (T,size,size,3) uint8, resized to the given size (as the lab's preprocess_video)."""
    return np.stack([np.asarray(Image.fromarray(image).resize(tuple(size), Image.BILINEAR)) for image in images])


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    parser = argparse.ArgumentParser(description="Convert recorded demos to a LeRobot dataset for pi0.5.")
    parser.add_argument("tasks", nargs="+", choices=list(cfg["tasks"]), help="task names (demo/data/<task>/)")
    parser.add_argument("--num-demos", type=int, help="use only the first N demos of each task")
    args = parser.parse_args()
    views, size = cfg["lerobot"], cfg["lerobot"]["image_size"]

    files = demo_files(cfg, args.tasks, args.num_demos)
    if not files:
        parser.error(f"no demos in {cfg['save_dir']}/{{{','.join(args.tasks)}}}/ (record them with demo/get_demo.py)")
    with h5py.File(files[0], "r") as f:
        fps = int(f.attrs["fps"])
    task_name, num_demos = "_".join(args.tasks), len(files)
    output_path = HF_LEROBOT_HOME / f"{task_name}_{num_demos}"
    if output_path.exists():
        shutil.rmtree(output_path)
    print(f"[INFO] {num_demos} demos at {fps} Hz -> {output_path}")

    dataset = LeRobotDataset.create(
        repo_id = f"{task_name}_{num_demos}",
        robot_type = "panda",
        # The recorded rate (record_freq): the rate the robot is commanded at.
        fps=fps,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (size[1], size[0], 3),
                "names": ["height", "width", "channel"],
            },
            # Kept for compatibility with the official DROID LeRobot schema.
            # pi05_droid uses the first exterior camera and masks this view.
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (size[1], size[0], 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (size[1], size[0], 3),
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

    for file in files:
        print(f"[Processing] {file}")
        with h5py.File(file, "r") as f:
            if int(f.attrs["fps"]) != fps:
                raise SystemExit(f"[ERROR] {file.name} was recorded at {f.attrs['fps']} Hz, the others at {fps} Hz")
            data = {}
            data["joint_vel"] = f["actions/joint_vel"][:]
            data["joint_pos"] = f["obs/joint_state"][:]
            data["exterior_1"] = resize(f[f"obs/{views['exterior_image_1_left']}"][:], size)
            data["exterior_2"] = (resize(f[f"obs/{views['exterior_image_2_left']}"][:], size)
                                  if views["exterior_image_2_left"] else None)
            data["wrist"] = resize(f[f"obs/{views['wrist_image_left']}"][:], size)
            data["gripper_state"] = f["obs/gripper_state"][:]                 # 1 = closed, 0 = open (DROID)
            data["gripper_action"] = f["actions/gripper_action"][:]
            data["task_description"] = str(f.attrs["task_description"])
        assert data["joint_vel"].shape[1] == 7, "Expected joint_vel to have shape (T, 7)"
        print(f"Loaded data with {len(data['exterior_1'])} frames. Task description: {data['task_description']}")

        for step in range(len(data["exterior_1"])):
            dataset.add_frame(
                {
                    "exterior_image_1_left": data["exterior_1"][step],
                    # Black placeholder if no second exterior camera is configured.
                    "exterior_image_2_left": (data["exterior_2"][step] if data["exterior_2"] is not None
                                              else np.zeros_like(data["exterior_1"][step])),
                    "wrist_image_left": data["wrist"][step],
                    "joint_position": np.array(data["joint_pos"][step], dtype="float32"),
                    "gripper_position": np.asarray(data["gripper_state"][step], dtype="float32").reshape(1),
                    "actions": np.append(np.array(data["joint_vel"][step], dtype="float32"), data["gripper_action"][step]).astype("float32"),
                    "task": data["task_description"]
                }
            )
        dataset.save_episode()


if __name__ == "__main__":
    main()
