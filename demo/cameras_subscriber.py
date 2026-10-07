"""Newest frame of each camera topic published by demo/cameras_publisher.py (or the lab's codata publisher).

Messages are multipart [topic, header + raw bytes]: <name>_img_rgb has header >III (h, w, c) and uint8 pixels,
<name>_img_depth has header >II (h, w) and uint16 depth.
"""

import struct
import threading
import time

import numpy as np
import zmq


class CamerasSubscriber:
    def __init__(self, topics, server_addr="localhost", port=8082):
        self.topics = list(topics)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{server_addr}:{port}")
        for topic in self.topics:
            self._socket.setsockopt(zmq.SUBSCRIBE, topic.encode())
        self._frames, self._times = {}, {}
        self._lock = threading.Lock()
        self._done = False
        self._thread = None

    def start_thread(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._done:
            if not poller.poll(100):
                continue
            topic, message = self._socket.recv_multipart()
            topic = topic.decode()
            if topic not in self.topics:                 # subscriptions match prefixes: keep exact names only
                continue
            depth = topic.endswith("_img_depth")
            size = 8 if depth else 12
            shape = struct.unpack(">II" if depth else ">III", message[:size])
            frame = np.frombuffer(message, dtype=np.uint16 if depth else np.uint8, offset=size).reshape(shape)
            with self._lock:
                self._frames[topic], self._times[topic] = frame, time.monotonic()

    def get_last_frames(self):
        """{topic: newest frame} for the topics received so far."""
        with self._lock:
            return dict(self._frames)

    def ages(self):
        """{topic: seconds since its newest frame}."""
        now = time.monotonic()
        with self._lock:
            return {topic: now - t for topic, t in self._times.items()}

    def wait_for_topics(self, timeout=10.0):
        """Wait until every topic has a frame; returns the topics still missing."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            missing = [topic for topic in self.topics if topic not in self.get_last_frames()]
            if not missing:
                return []
            time.sleep(0.1)
        return missing

    def close_subscriber(self):
        self._done = True
        if self._thread is not None:
            self._thread.join()
        self._socket.close()
        self._context.term()
