import argparse
from datetime import datetime
from pathlib import Path
import re
import select
import sys
import threading
import time

import numpy as np
from termcolor import colored
import yaml

try:
   import cv2
   import zmq
except ImportError:
   cv2 = None
   zmq = None

try:
   from .franka import Franka
except ImportError:  # Supports copying real/ to a separate computer.
   from franka import Franka


LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def load_config(path):
   with open(path) as file:
       config = yaml.safe_load(file)
   task_dir_raw = config["trajectory"]["task_dir"]
   task_dir = Path(task_dir_raw).expanduser()
   if not task_dir.is_absolute():
       task_dir = (Path(__file__).resolve().parents[3] / task_dir).resolve()
   if not task_dir.exists():
       candidate = Path(__file__).resolve().parents[3] / "dataset" / "trajectories" / task_dir_raw
       if candidate.exists():
           task_dir = candidate.resolve()
   if not task_dir.exists() and (task_dir.parent / "trajectories" / task_dir.name).exists():
       task_dir = task_dir.parent / "trajectories" / task_dir.name
   config["task_dir"] = task_dir
   file_spec = config.get("trajectory", {}).get("file")
   if file_spec:
       trajectory_path = Path(file_spec)
       if not trajectory_path.is_absolute():
           trajectory_path = task_dir / trajectory_path
       config["trajectory_path"] = trajectory_path
   else:
       config["trajectory_path"] = None

   grasp_file = config.get("grasp", {}).get("file", "grasp_trajectory.npz")
   grasp_path = Path(grasp_file)
   if not grasp_path.is_absolute():
       grasp_path = task_dir / grasp_path
   config["grasp_path"] = grasp_path
   return config


def get_task_prefix(task_dir: Path) -> str:
   """Extract a descriptive task prefix from task_dir, including model name if nested under trajectories.

   For example:
     dataset/trajectories/wan2_2/bowl_ours -> 'wan2_2_bowl_ours'
     dataset/trajectories/ltx_2_3/bowl_ours -> 'ltx_2_3_bowl_ours'
     dataset/trajectories/cogvideox/bowl_ours -> 'cogvideox_bowl_ours'
     dataset/trajectories/move_banana_in_pan_ours -> 'move_banana_in_pan_ours'
   """
   resolved = task_dir.resolve()
   parts = list(resolved.parts)
   if "trajectories" in parts:
       idx = len(parts) - 1 - parts[::-1].index("trajectories")
       rel_parts = parts[idx + 1 :]
       if rel_parts:
           return "_".join(rel_parts)
   return task_dir.name


def quat2rot(q):
   """Convert wxyz quaternion to 3x3 rotation matrix."""
   w, x, y, z = q
   return np.array([
       [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
       [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
       [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
   ], dtype=np.float64)


def rot_error(R_target, R_curr):
   """Compute 3D orientation error vector (axis * angle) from R_curr to R_target."""
   R_err = R_target @ R_curr.T
   v = 0.5 * np.array([
       R_err[2, 1] - R_err[1, 2],
       R_err[0, 2] - R_err[2, 0],
       R_err[1, 0] - R_err[0, 1],
   ], dtype=np.float64)
   tr = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
   theta = np.arccos(tr)
   sin_theta = np.sin(theta)
   if sin_theta > 1e-4:
       return v * (theta / sin_theta)
   return v


def interpolate_tcp(trajectory, sample):
   """Interpolate TCP trajectory [x, y, z, qw, qx, qy, qz] with normalized quaternion lerp."""
   lower = min(int(sample), len(trajectory) - 1)
   upper = min(lower + 1, len(trajectory) - 1)
   fraction = sample - lower
   pos = trajectory[lower, :3] * (1.0 - fraction) + trajectory[upper, :3] * fraction
   q_low = trajectory[lower, 3:]
   q_up = trajectory[upper, 3:]
   if np.dot(q_low, q_up) < 0.0:
       q_up = -q_up
   q = q_low * (1.0 - fraction) + q_up * fraction
   norm = np.linalg.norm(q)
   if norm > 1e-12:
       q = q / norm
   return np.concatenate([pos, q])


def load_trajectory(path, require_feasible=True):
   """Load trajectory from npz, auto-detecting 'joint_trajectory' or 'tcp_trajectory'."""
   with np.load(path, allow_pickle=False) as archive:
       if require_feasible and "feasible" in archive and not bool(archive["feasible"]):
           raise ValueError("Trajectory is marked infeasible")
       if "joint_trajectory" in archive:
           trajectory = np.asarray(archive["joint_trajectory"], dtype=np.float64)
           traj_type = "joint"
       elif "tcp_trajectory" in archive:
           trajectory = np.asarray(archive["tcp_trajectory"], dtype=np.float64)
           traj_type = "tcp"
           print(colored(f">> [SIGNAL] Reading TCP trajectory (positions and orientations) from {Path(path).name}", "yellow"))
       else:
           raise KeyError(f"{path} does not contain joint_trajectory or tcp_trajectory")

   if trajectory.ndim != 2 or trajectory.shape[1] != 7 or len(trajectory) < 2:
       raise ValueError(f"Expected trajectory shape (T, 7), got {trajectory.shape}")
   if not np.all(np.isfinite(trajectory)):
       raise ValueError("Trajectory contains non-finite values")

   if traj_type == "joint":
       if np.any(trajectory < LOWER) or np.any(trajectory > UPPER):
           raise ValueError("Trajectory exceeds the stock Panda joint limits")
   else:
       quats = trajectory[:, 3:]
       norms = np.linalg.norm(quats, axis=1, keepdims=True)
       if np.any(norms < 1e-4):
           raise ValueError("TCP trajectory contains zero-norm quaternion")
       trajectory = trajectory.copy()
       trajectory[:, 3:] /= norms

   return trajectory, traj_type


def load_joint_trajectory(path, require_feasible=True):
   trajectory, traj_type = load_trajectory(path, require_feasible=require_feasible)
   if traj_type != "joint":
       raise ValueError(f"Expected joint trajectory in {path}, got {traj_type}")
   return trajectory


def interpolate(trajectory, sample):
   lower = min(int(sample), len(trajectory) - 1)
   upper = min(lower + 1, len(trajectory) - 1)
   fraction = sample - lower
   return trajectory[lower] * (1.0 - fraction) + trajectory[upper] * fraction


def playback_rate(trajectory, source_hz, control, traj_type="joint"):
   requested = float(control["playback_rate"])
   if requested <= 0.0:
       raise ValueError("control.playback_rate must be positive")
   if not control["automatic_slowdown"]:
       return requested
   if traj_type == "tcp":
       sampled_speed = np.max(
           np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1) * source_hz
       )
       cart_limit = float(control.get("cartesian_velocity_limit", 0.25))
       safe_speed = cart_limit * float(control["speed_fraction"])
   else:
       sampled_speed = np.max(
           np.linalg.norm(np.diff(trajectory, axis=0), axis=1) * source_hz
       )
       safe_speed = float(control["velocity_limit"]) * float(control["speed_fraction"])
   return min(requested, safe_speed / max(sampled_speed, 1e-12))


def list_available_demos(task_dir: Path):
   """Return a sorted list of demo file names found in task_dir."""
   demos = []
   for p in task_dir.glob("demo_*.npz"):
       if p.name == "grasp_trajectory.npz":
           continue
       demos.append(p.name)

   def _sort_key(name: str):
       m = re.search(r"demo_(\d+)", name)
       return (int(m.group(1)), name) if m else (999999, name)

   return sorted(demos, key=_sort_key)


def parse_demo_query(demo_input: str):
   """
   Parse input string like '1', '1 2', '1, 2', '1:2', '1 s2', '1 sample 2', '1 0 2'.
   Returns (demo_num, sample_num, seed_num).
   """
   s = demo_input.strip()
   m_full = re.search(r"demo_(\d+)(?:_seed_(\d+))?(?:_sample_(\d+))?", s)
   if m_full and ("seed" in s or "sample" in s):
       d = int(m_full.group(1)) if m_full.group(1) else None
       sd = int(m_full.group(2)) if m_full.group(2) else None
       sp = int(m_full.group(3)) if m_full.group(3) else None
       return d, sp, sd

   cleaned = s.replace("demo_", "").replace("demo", "").replace(".npz", "")
   tokens = re.split(r"[\s,:/]+", cleaned.strip())
   tokens = [t for t in tokens if t]

   demo_num = None
   sample_num = None
   seed_num = None

   i = 0
   nums = []
   while i < len(tokens):
       t = tokens[i].lower()
       if t in ("sample", "s") and i + 1 < len(tokens) and tokens[i + 1].isdigit():
           sample_num = int(tokens[i + 1])
           i += 2
       elif t.startswith("sample") and t[6:].isdigit():
           sample_num = int(t[6:])
           i += 1
       elif t.startswith("s") and t[1:].isdigit():
           sample_num = int(t[1:])
           i += 1
       elif t == "seed" and i + 1 < len(tokens) and tokens[i + 1].isdigit():
           seed_num = int(tokens[i + 1])
           i += 2
       elif t.startswith("seed") and t[4:].isdigit():
           seed_num = int(t[4:])
           i += 1
       elif t.isdigit():
           nums.append(int(t))
           i += 1
       else:
           i += 1

   if nums:
       if demo_num is None:
           demo_num = nums[0]
       if len(nums) == 2 and sample_num is None:
           sample_num = nums[1]
       elif len(nums) >= 3:
           if seed_num is None:
               seed_num = nums[1]
           if sample_num is None:
               sample_num = nums[2]

   return demo_num, sample_num, seed_num


def resolve_demo_trajectory(task_dir: Path, demo_input: str) -> Path:
   """Resolve user input (e.g. '1', '1 2', 'demo_1', 'demo_1_seed_0.npz') to a file path in task_dir."""
   demo_input = demo_input.strip()
   if not demo_input:
       raise ValueError("Demo identifier cannot be empty")

   exact_path = task_dir / demo_input
   if exact_path.is_file():
       return exact_path
   if (task_dir / f"{demo_input}.npz").is_file():
       return task_dir / f"{demo_input}.npz"

   all_demo_files = [p for p in task_dir.glob("demo_*.npz") if p.name != "grasp_trajectory.npz" and p.is_file()]
   if not all_demo_files:
       raise FileNotFoundError(f"No demo trajectories found in {task_dir}")

   demo_num, sample_num, seed_num = parse_demo_query(demo_input)

   if demo_num is not None:
       matching_files = []
       for p in all_demo_files:
           m_d = re.search(r"demo_(\d+)", p.name)
           if not m_d or int(m_d.group(1)) != demo_num:
               continue

           m_seed = re.search(r"seed_(\d+)", p.name)
           m_samp = re.search(r"sample_(\d+)", p.name)

           f_seed = int(m_seed.group(1)) if m_seed else None
           f_samp = int(m_samp.group(1)) if m_samp else None

           if seed_num is not None and f_seed is not None and f_seed != seed_num:
               continue

           if sample_num is not None:
               if f_samp is not None:
                   if f_samp != sample_num:
                       continue
               elif f_seed is not None and seed_num is None:
                   if f_seed != sample_num:
                       continue

           matching_files.append((f_samp if f_samp is not None else 0, f_seed if f_seed is not None else 0, p))

       matching_files.sort(key=lambda item: (item[0], item[1], item[2].name))

       if matching_files:
           selected = matching_files[0][2]
           if len(matching_files) > 1:
               if sample_num is None:
                   samples_list = [f"{item[0]:02d}" for item in matching_files]
                   print(
                       colored(
                           f"Demo {demo_num} has {len(matching_files)} samples ({', '.join(samples_list[:6])}...). "
                           f"Defaulting to {selected.name}.\n"
                           f"Tip: to pick a specific sample, enter '{demo_num} <sample>' (e.g. '{demo_num} {samples_list[min(1, len(samples_list)-1)]}')",
                           "cyan",
                       )
                   )
               else:
                   print(f"Multiple matches for demo {demo_num} sample {sample_num}, using: {selected.name}")
           else:
               if sample_num is not None:
                   print(f"Selected demo {demo_num}, sample {sample_num}: {selected.name}")
           return selected

       if sample_num is not None:
           demo_only = [p for p in all_demo_files if (re.search(r"demo_(\d+)", p.name) and int(re.search(r"demo_(\d+)", p.name).group(1)) == demo_num)]
           if demo_only:
               avail_samp = sorted(list({int(re.search(r"sample_(\d+)", p.name).group(1)) for p in demo_only if re.search(r"sample_(\d+)", p.name)}))
               samp_str = ", ".join(str(s) for s in avail_samp) if avail_samp else "none"
               raise FileNotFoundError(
                   f"Sample {sample_num} not found for demo {demo_num}. Available samples for demo {demo_num}: {samp_str}"
               )

   clean_input = demo_input.replace("demo_", "").replace(".npz", "")
   candidates = []
   for p in sorted(task_dir.glob(f"demo_{clean_input}.npz")):
       if p.is_file() and p not in candidates:
           candidates.append(p)
   for p in sorted(task_dir.glob(f"demo_{clean_input}_*.npz")):
       if p.is_file() and p not in candidates:
           candidates.append(p)

   if not candidates:
       available = list_available_demos(task_dir)
       avail_str = ", ".join(p.replace("demo_", "").replace(".npz", "") for p in available[:12])
       if len(available) > 12:
           avail_str += f", ... ({len(available)} total)"
       raise FileNotFoundError(
           f"No trajectory matching '{demo_input}' in {task_dir}.\nAvailable demos: {avail_str}"
       )

   if len(candidates) > 1:
       print(f"Multiple matches found for '{demo_input}', using: {candidates[0].name}")
   return candidates[0]


def prompt_with_camera(prompt_text: str, camera_feed=None) -> str:
   """Prompt user on terminal while keeping camera window responsive."""
   print(prompt_text, end="", flush=True)
   while True:
       if camera_feed is not None:
           camera_feed.update()
       r, _, _ = select.select([sys.stdin], [], [], 0.03)
       if r:
           line = sys.stdin.readline()
           return line.strip()


class CameraFeed:
   """Non-blocking background camera feed client with optional video recording."""

   def __init__(self, address: str = "tcp://127.0.0.1:5555", window_name: str = "Robot Camera"):
       self.address = address
       self.window_name = window_name
       self.available = False
       self.running = False
       self.thread = None
       self.latest_frame = None
       self.lock = threading.Lock()
       self.recording = False
       self.record_path = None
       self.record_fps = 30.0
       self.video_writer = None

       if cv2 is None or zmq is None:
           print("[camera] opencv-python or pyzmq not available; skipping camera feed.")
           return

       try:
           ctx = zmq.Context()
           sock = ctx.socket(zmq.REQ)
           sock.setsockopt(zmq.RCVTIMEO, 2500)
           sock.setsockopt(zmq.SNDTIMEO, 1500)
           sock.connect(self.address)
           sock.send(b"info")
           info = sock.recv_json()
           sock.close(linger=0)
           ctx.term()
           if info.get("ok"):
               self.available = True
       except Exception:
           self.available = False

       if not self.available:
           print(f"[camera] Camera server not reachable on {self.address}; skipping camera feed.")
           return

       print(f"[camera] Connected to camera server on {self.address}. Displaying camera feed...")
       self.running = True
       self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
       self.thread.start()
       # Wait up to 1.5s for the first frame to arrive and display it immediately
       deadline = time.monotonic() + 1.5
       while time.monotonic() < deadline and self.latest_frame is None and self.running:
           time.sleep(0.03)
       if self.latest_frame is not None:
           self.update()

   def _fetch_loop(self):
       ctx = zmq.Context()

       def _create_socket():
           s = ctx.socket(zmq.REQ)
           s.setsockopt(zmq.RCVTIMEO, 2500)
           s.setsockopt(zmq.SNDTIMEO, 2500)
           s.connect(self.address)
           return s

       sock = _create_socket()
       try:
           while self.running:
               try:
                   sock.send(b"capture")
                   parts = sock.recv_multipart()
                   if len(parts) >= 2:
                       bgr = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_COLOR)
                       if bgr is not None:
                           with self.lock:
                               self.latest_frame = bgr
                               if self.recording and self.record_path is not None:
                                   if self.video_writer is None:
                                       h, w = bgr.shape[:2]
                                       fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                       self.video_writer = cv2.VideoWriter(
                                           str(self.record_path), fourcc, float(self.record_fps), (w, h)
                                       )
                                   if self.video_writer.isOpened():
                                       self.video_writer.write(bgr)
               except Exception:
                   try:
                       sock.close(linger=0)
                   except Exception:
                       pass
                   sock = _create_socket()
                   time.sleep(0.05)
       finally:
           if sock is not None:
               sock.close(linger=0)
           ctx.term()

   def start_recording(self, output_path: Path, fps: float = 30.0):
       output_path = Path(output_path).resolve()
       output_path.parent.mkdir(parents=True, exist_ok=True)
       with self.lock:
           self.record_path = output_path
           self.record_fps = fps
           self.recording = True
       print(colored(f"[camera] Started video recording to: {output_path}", "cyan"), flush=True)

   def stop_recording(self):
       writer = None
       saved_path = None
       with self.lock:
           if self.recording:
               self.recording = False
               writer = self.video_writer
               self.video_writer = None
               saved_path = self.record_path
               self.record_path = None
       if writer is not None:
           writer.release()
       if saved_path is not None:
           print(colored(f"[camera] Saved video recording to: {saved_path}", "cyan"), flush=True)

   def update(self):
       if not self.available or not self.running:
           return
       frame = None
       with self.lock:
           if self.latest_frame is not None:
               frame = self.latest_frame.copy()
       if frame is not None:
           try:
               cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
               cv2.imshow(self.window_name, frame)
               key = cv2.waitKey(1) & 0xFF
               if key == ord("q") or key == 27:
                   raise KeyboardInterrupt
           except KeyboardInterrupt:
               raise
           except Exception:
               pass
       else:
           try:
               key = cv2.waitKey(1) & 0xFF
               if key == ord("q") or key == 27:
                   raise KeyboardInterrupt
           except KeyboardInterrupt:
               raise
           except Exception:
               pass

   def close(self):
       if not self.available:
           return
       self.stop_recording()
       self.running = False
       if self.thread is not None:
           self.thread.join(timeout=1.0)
       try:
           cv2.destroyAllWindows()
       except Exception:
           pass

   def __enter__(self):
       return self

   def __exit__(self, exc_type, exc_val, exc_tb):
       self.close()


def sleep_with_camera(duration, camera_feed=None,
