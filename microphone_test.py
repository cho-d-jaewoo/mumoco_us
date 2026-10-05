"""Check the microphone and Whisper: START (joystick) or Enter starts recording, START / Enter again
stops it and prints what was said, with segment and word times on one session timeline.
Microphone (MIC_NAME), Whisper model and language: mumoco/config.py. Ctrl+C quits.
"""

import time

from mumoco.config import WHISPER_MODEL
from mumoco.joystick_input import open_joystick
from mumoco.microphone import Microphone, Transcriber, audio_level_db
from mumoco.utils import read_terminal_line


def wait_for_toggle(joystick):
    """Joystick START or Enter."""
    while not ((joystick and "START" in joystick.poll()) or read_terminal_line() is not None):
        time.sleep(0.02)


def main():
    mic = Microphone()
    print(f"[INFO] Loading Whisper model '{WHISPER_MODEL}' (first run downloads it)...")
    transcriber = Transcriber()
    print(f"[INFO] Whisper ready on {transcriber.model.device}.")
    joystick = open_joystick()
    session_t0 = time.monotonic()                        # all times below: [s] since program start
    try:
        while True:
            print("\nPress START / Enter to start recording (Ctrl+C to quit).")
            wait_for_toggle(joystick)
            mic.start()
            print(f"[REC] {time.monotonic() - session_t0:7.2f} s  Recording... speak now. START / Enter to stop.")
            wait_for_toggle(joystick)
            recording = mic.stop()
            if recording is None:
                print("[WARNING] No audio captured.")
                continue
            t_start = recording["t_start"] - session_t0
            duration = len(recording["audio"]) / recording["sample_rate"]
            level = audio_level_db(recording["audio"])
            print(f"[REC] {t_start:7.2f} s - {t_start + duration:7.2f} s  ({duration:.1f} s, peak {level:.0f} dBFS)")
            for warning in recording["warnings"]:
                print(f"[WARNING] Audio: {warning}")
            if level < -50:
                print("[WARNING] Almost silent: check the microphone (MIC_NAME in mumoco/config.py).")

            t = time.monotonic()
            result = transcriber.transcribe(recording)
            print(f"[INFO] Transcribed in {time.monotonic() - t:.2f} s.")
            print(f'\n>>> "{result["text"]}"\n')
            for s in result["segments"]:
                print(f"  {s['start'] - session_t0:7.2f} - {s['end'] - session_t0:7.2f} s  {s['text']}")
                print("      " + " ".join(f"{w['word']}@{w['start'] - session_t0:.2f}" for w in s["words"]))
    except KeyboardInterrupt:
        print("\n[INFO] Quit.")
    finally:
        mic.close()
        if joystick:
            joystick.close()


if __name__ == "__main__":
    main()
