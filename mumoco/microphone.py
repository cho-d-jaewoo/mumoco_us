"""Microphone recording and Whisper speech-to-text (from the lab's deploy.py VoiceTaskRecorder).

All times are time.monotonic() seconds, the clock Robot.guide() stamps arm states with, so spoken
and physical corrections can later be put on one timeline.

Run `python3 microphone_test.py` (from the repository root) to check the microphone and Whisper.
"""

import time

import numpy as np

from .config import MIC_SAMPLE_RATE, WHISPER_LANGUAGE, WHISPER_MODEL


class Microphone:
    """start() opens the input stream, stop() closes it and returns the recording:
    {"t_start": monotonic time of the first sample, "sample_rate", "audio": float32 mono, "warnings"}."""

    def __init__(self, device=None, sample_rate=MIC_SAMPLE_RATE):
        import sounddevice as sd                         # here, so robot-only scripts do not need it
        self.sd = sd
        self.device, self.sample_rate = device, sample_rate
        self.stream = None
        self.chunks, self.warnings, self.t_start = [], [], None

    @property
    def recording(self):
        return self.stream is not None

    def _callback(self, indata, frames, _time_info, status):   # PortAudio thread
        if self.t_start is None:                         # this block was captured during the last frames/rate s
            self.t_start = time.monotonic() - frames / self.sample_rate
        if status:
            self.warnings.append(str(status))
        self.chunks.append(indata[:, 0].copy())

    def start(self):
        if self.recording:
            return
        self.chunks, self.warnings, self.t_start = [], [], None
        self.stream = self.sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                                          device=self.device, callback=self._callback)
        self.stream.start()

    def stop(self):
        """The recording, or None if no audio was captured."""
        if not self.recording:
            return None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        if not self.chunks:
            return None
        return {"t_start": self.t_start, "sample_rate": self.sample_rate,
                "audio": np.concatenate(self.chunks), "warnings": self.warnings}

    def close(self):
        self.stop()


def audio_level_db(audio):
    """Peak level [dBFS]; around -60 or lower means the microphone heard (almost) nothing."""
    return 20 * np.log10(max(float(np.max(np.abs(audio))), 1e-10))


class Transcriber:
    """Whisper speech-to-text. transcribe(recording) returns {"text", "segments"}; each segment is
    {"start", "end", "text", "words": [{"start", "end", "word"}]} with start/end in the recording's
    monotonic clock."""

    def __init__(self, model_name=WHISPER_MODEL, language=WHISPER_LANGUAGE):
        import whisper
        self.model = whisper.load_model(model_name)
        self.language = language
        # deploy.py: fp16 halves the transcription time on CUDA; CPU has no fp16 kernels.
        self.fp16 = self.model.device.type == "cuda"

    def transcribe(self, recording):
        if recording["sample_rate"] != 16_000:
            raise ValueError("Whisper needs 16 kHz audio.")
        result = self.model.transcribe(recording["audio"], language=self.language, fp16=self.fp16,
                                       word_timestamps=True)
        t0 = recording["t_start"]
        segments = [{"start": float(t0 + s["start"]), "end": float(t0 + s["end"]), "text": s["text"].strip(),
                     "words": [{"start": float(t0 + w["start"]), "end": float(t0 + w["end"]), "word": w["word"].strip()}
                               for w in s.get("words", [])]}
                    for s in result["segments"]]
        return {"text": result["text"].strip(), "segments": segments}
