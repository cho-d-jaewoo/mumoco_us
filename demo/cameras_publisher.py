"""Publish the three RealSense cameras over ZMQ (the lab's codata cameras_publisher.py with all_realsense, standalone).

    python3 -m demo.cameras_publisher           # fps, port: demo/config/cameras_publisher.yaml

Cameras as the lab's gripper/right/left_realsense configs: 640x360 color + depth, the DataCollectionSettings.json
preset, IR emitter on for left/right. Each camera is read on its own thread; the publisher sends the newest frame
of every camera at `fps` as topic <name>_img_rgb (header >III h, w, c + uint8 RGB) and <name>_img_depth
(header >II h, w + uint16 depth aligned to the color image).
"""

import struct
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

CONFIG = Path(__file__).parent / "config" / "cameras_publisher.yaml"
JSON_SETTINGS = Path(__file__).parent / "config" / "DataCollectionSettings.json"
WIDTH, HEIGHT = 640, 360                                  # codata img_shape [360, 640]
CAMERAS = {                                               # name: (serial number, IR emitter; None = preset)
    "gripper": ("332522073288", None),
    "right_realsense": ("153122073885", True),
    "left_realsense": ("233622075120", True),
}


class RealSense:
    """Color and depth aligned to it of one camera, with the DataCollectionSettings.json preset."""

    def __init__(self, serial, fps=30, ir_emitter=None):
        import pyrealsense2 as rs
        self.pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, fps)
        device = self.pipeline.start(config).get_device()
        advanced = rs.rs400_advanced_mode(device)
        if not advanced.is_enabled():
            raise SystemExit(f"[ERROR] Advanced mode is off on camera {serial}: enable it in realsense-viewer")
        advanced.load_json(JSON_SETTINGS.read_text())
        if ir_emitter is not None:
            device.first_depth_sensor().set_option(rs.option.emitter_enabled, 1.0 if ir_emitter else 0.0)
        self.align = rs.align(rs.stream.color)
        for _ in range(30):                              # discard frames while the settings take effect
            self.pipeline.wait_for_frames()
        self.frame, self.done = None, False

    def _run(self):
        while not self.done:
            frames = self.align.process(self.pipeline.wait_for_frames())
            self.frame = {"img_rgb": np.asanyarray(frames.get_color_frame().get_data()).copy(),
                          "img_depth": np.asanyarray(frames.get_depth_frame().get_data()).copy()}

    def get_frame(self):
        return self.frame

    def close_camera(self):
        self.done = True


class CamerasPublisher:
    def __init__(self, fps=30, port=8082):
        self.send_period = 1.0 / fps
        self.cameras_obj, self.cameras_thread = {}, {}
        for name, (serial, ir_emitter) in CAMERAS.items():
            print(f"[*] Initializing {name} camera")
            camera = RealSense(serial, fps, ir_emitter)
            thread = threading.Thread(target=camera._run, daemon=True)
            thread.start()
            self.cameras_obj[name], self.cameras_thread[name] = camera, thread
        time.sleep(2)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://*:{port}")
        print(f"[*] Publishing {', '.join(self.cameras_obj)} on port {port}")

    def start_publisher(self):
        try:
            next_time = time.monotonic()
            while True:
                for name, camera in self.cameras_obj.items():
                    frame = camera.get_frame()
                    if frame is None:
                        continue
                    h, w, c = frame["img_rgb"].shape
                    self._socket.send_multipart([f"{name}_img_rgb".encode(),
                                                 struct.pack(">III", h, w, c) + frame["img_rgb"].tobytes()])
                    h, w = frame["img_depth"].shape
                    self._socket.send_multipart([f"{name}_img_depth".encode(),
                                                 struct.pack(">II", h, w) + frame["img_depth"].tobytes()])
                next_time += self.send_period
                time.sleep(max(0.0, next_time - time.monotonic()))
        except KeyboardInterrupt:
            print("[*] Interrupted. Closing cameras and publisher")
        finally:
            self.close_cameras()
            self._socket.close()
            self._context.term()

    def close_cameras(self):
        for camera in self.cameras_obj.values():
            camera.close_camera()
        for thread in self.cameras_thread.values():
            thread.join()
        for camera in self.cameras_obj.values():
            camera.pipeline.stop()


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    CamerasPublisher(cfg["fps"], cfg["port"]).start_publisher()


if __name__ == "__main__":
    main()
