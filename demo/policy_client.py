"""Client of an openpi policy server (scripts/serve_policy.py), queried in the background as the lab's pi0.py.

Protocol as openpi_client (websocket, msgpack with numpy arrays): the server first sends its metadata, then
answers every observation with {"actions": (horizon, 8), ...}.

The control loop posts an observation every tick and takes one action per tick. Inference runs back to back
in a thread, always on the newest observation, and each new action chunk replaces what is left of the
previous one. So the robot keeps moving on the old chunk while the next one is computed.
"""

import functools
import threading
import time

import msgpack
import numpy as np
import websockets.sync.client


def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class PolicyClient:
    def __init__(self, host="localhost", port=9000):
        self.uri = f"ws://{host}:{port}"
        self._packer = msgpack.Packer(default=pack_array)
        self._ws, self.metadata = self._connect()
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._observation, self._actions, self._index = None, [], 0
        self._generation = 0                 # clear() raises it: a chunk computed before that is dropped
        self._done = False
        self.error = None                    # exception that ended the inference thread
        self.chunks, self.latencies = 0, []  # action chunks received / [s] each took (since the last clear)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _connect(self):
        """Wait for the server (Ctrl+C gives up): it may still be loading the model, or the tunnel is not up."""
        waiting = False
        while True:
            try:
                ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
                return ws, unpackb(ws.recv())
            except (OSError, websockets.exceptions.WebSocketException, EOFError) as e:
                if not waiting:
                    print(f"[INFO] Waiting for the policy server at {self.uri} ({type(e).__name__})...")
                    waiting = True
                time.sleep(3)

    def _run(self):
        while not self._done:
            if not self._new.wait(0.1):
                continue
            with self._lock:
                observation, self._observation = self._observation, None
                generation = self._generation
                self._new.clear()
            if observation is None:
                continue
            start = time.monotonic()
            try:
                self._ws.send(self._packer.pack(observation))
                response = self._ws.recv()
                if isinstance(response, str):            # the server sends its traceback as text
                    raise RuntimeError(f"Error in the policy server:\n{response}")
                actions = np.asarray(unpackb(response)["actions"], dtype=np.float64)
            except Exception as e:
                self.error = e
                return
            with self._lock:
                if generation == self._generation:
                    self._actions, self._index = list(actions), 0
                    self.chunks += 1
                    self.latencies.append(time.monotonic() - start)

    def post(self, observation):
        """The newest observation; replaces one that inference has not started on yet."""
        with self._lock:
            self._observation = observation
        self._new.set()

    def next_action(self):
        """(action (8,), chunk number, index in the chunk), or None while no action is buffered."""
        with self._lock:
            if not self._actions:
                return None
            self._index += 1
            return self._actions.pop(0), self.chunks, self._index - 1

    def clear(self):
        """Drop the pending observation, the buffered actions and the statistics (before a new rollout)."""
        with self._lock:
            self._observation, self._actions = None, []
            self._generation += 1
            self._new.clear()
            self.chunks, self.latencies = 0, []

    def close(self):
        self._done = True
        self._thread.join(timeout=2)
        try:
            self._ws.close()
        except Exception:
            pass
