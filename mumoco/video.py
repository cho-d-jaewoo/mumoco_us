"""Synchronized looping playback of task videos (animated WebP) inside the Tk window.

Each video is decoded and resized by its own worker thread; the window thread only shows ready frames.
Every loop starts for all videos at the same time: a video that ends first holds its last frame until
the others have ended too, then all restart together.
"""

import queue
import threading
import time

END = None                          # queued after the last frame of a loop


class VideoStream:
    """Decodes one loop of the video at a time into a small queue of (frame, duration in ms)."""

    def __init__(self, path, size):
        self.path, self.size, self.error = path, size, None
        self.frames = queue.Queue(maxsize=8)
        self._restart, self._stop = threading.Event(), threading.Event()
        threading.Thread(target=self._decode, daemon=True).start()

    def _decode(self):
        from PIL import Image
        try:
            with Image.open(self.path) as video:
                while True:
                    for i in range(getattr(video, "n_frames", 1)):
                        video.seek(i)
                        frame = video.convert("RGB").resize(self.size, Image.BILINEAR)
                        if not self._put((frame, video.info.get("duration", 50))):
                            return
                    if not self._put(END):
                        return
                    self._restart.wait()                 # next loop when all videos have ended
                    self._restart.clear()
                    if self._stop.is_set():
                        return
        except Exception as e:                           # missing, corrupted or unreadable file
            self.error = f"Cannot play {self.path.name}: {e}"
            self._put(END)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.frames.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def ready(self):
        return self.error is not None or not self.frames.empty()

    def restart(self):
        self._restart.set()

    def stop(self):
        self._stop.set()
        self._restart.set()


class SyncedVideos:
    """Plays VideoStreams in Tk labels (window thread only). on_error(index, message) reports a video
    that cannot be played; it then counts as ended in every loop."""

    def __init__(self, root, labels, paths, size, on_error):
        from PIL import ImageTk
        self.root, self.labels, self.on_error, self.ImageTk = root, labels, on_error, ImageTk
        self.streams = [VideoStream(path, size) for path in paths]
        self.reported, self.job = set(), None
        self._start_loop()

    def _start_loop(self):
        if not all(stream.ready() for stream in self.streams):     # synchronized start: all first frames ready
            self.job = self.root.after(10, self._start_loop)
            return
        self.start, self.due = time.monotonic(), [0.0] * len(self.streams)
        self.ended = [stream.error is not None and stream.frames.empty() for stream in self.streams]
        self._tick()

    def _tick(self):
        now = (time.monotonic() - self.start) * 1000.0
        for i, stream in enumerate(self.streams):
            while not self.ended[i] and now >= self.due[i] and not stream.frames.empty():
                item = stream.frames.get_nowait()
                if item is END:
                    self.ended[i] = True                 # hold the last frame
                    if stream.error and i not in self.reported:
                        self.reported.add(i)
                        self.on_error(i, stream.error)
                else:
                    frame, duration = item
                    image = self.ImageTk.PhotoImage(frame)
                    self.labels[i].configure(image=image)
                    self.labels[i].image = image         # keep a reference, or Tk shows nothing
                    self.due[i] += duration
        if all(self.ended):
            for stream in self.streams:
                stream.restart()
            self._start_loop()
        else:
            self.job = self.root.after(10, self._tick)

    def stop(self):
        if self.job is not None:
            self.root.after_cancel(self.job)
        for stream in self.streams:
            stream.stop()
