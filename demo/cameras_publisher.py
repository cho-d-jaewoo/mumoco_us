"""Publish the RealSense cameras over ZMQ (the lab's codata cameras_publisher.py with all_realsense, standalone).

    python3 -m demo.cameras_publisher           # settings: demo/config/cameras_publisher.yaml
    python3 -m demo.cameras_publisher --list    # serial numbers of the connected RealSense cameras

Each camera is read on its own thread; the publisher sends the newest frame of every camera at `fps` as
topic <name>_img_rgb (header >III h, w, c + uint8 RGB) and, with depth, <name>_img_depth (header >II h, w +
uint16 depth aligned to the color image).
"""

import argparse
import struct
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

CONFIG = Path(__file__).parent / "config" / "cameras_publisher.yaml"


class RealSense:
    def __init__(self, serial, width=640, height=480, fps=30, depth=True):
        import pyrealsense2 as rs
        self.pipeline, config = rs.pipeline(), rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.align = rs.align(rs.stream.color) if depth else None
        self.pipeline.start(config)
        for _ in range(30):                              # discard frames while auto exposure settles
            self.pipeline.wait_for_frames()
        self.frame, self.done = None, False

    def _run(self):
        while not self.done:
            frames = self.pipeline.wait_for_frames()
            if self.align:
                frames = self.align.process(frames)
            frame = {"img_rgb": np.asanyarray(frames.get_color_frame().get_data()).copy()}
            if self.align:
                frame["img_depth"] = np.asanyarray(frames.get_depth_frame().get_data()).copy()
            self.frame = frame

    def get_frame(self):
        return self.frame

    def close_camera(self):
        self.done = True


class CamerasPublisher:
    def __init__(self, cameras_conf, fps=30, port=8082):
        self.send_period = 1.0 / fps
        self.cameras_obj, self.cameras_thread = {}, {}
        for name, conf in cameras_conf.items():
            if not conf.get("serial"):
                raise SystemExit(f"[ERROR] No serial number for camera '{name}' in {CONFIG.name} "
                                 "(python3 -m demo.cameras_publisher --list)")
            print(f"[*] Initializing {name} camera")
            camera = RealSense(str(conf["serial"]), conf.get("width", 640), conf.get("height", 480), fps,
                               conf.get("depth", True))
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
                    if "img_depth" in frame:
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


def list_cameras():
    import pyrealsense2 as rs
    for device in rs.context().query_devices():
        print(f"{device.get_info(rs.camera_info.name)}: serial {device.get_info(rs.camera_info.serial_number)}")


def main():
    parser = argparse.ArgumentParser(description="Publish the RealSense cameras over ZMQ.")
    parser.add_argument("--list", action="store_true", help="print the serial numbers of the connected cameras")
    args = parser.parse_args()
    if args.list:
        list_cameras()
        return
    cfg = yaml.safe_load(CONFIG.read_text())
    CamerasPublisher(cfg["cameras"], cfg["fps"], cfg["port"]).start_publisher()


if __name__ == "__main__":
    main()
