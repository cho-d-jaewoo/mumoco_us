"""Check the microphone and Whisper: START (joystick) or Enter starts recording, START / Enter again
stops it and prints what was said, with segment and word times on one session timeline.

    python3 microphone_test.py                               # default microphone, Whisper turbo, English
    python3 microphone_test.py --list                        # list audio devices
    python3 microphone_test.py --device 3 --model small.en   # pick a microphone / a faster model
    python3 microphone_test.py --language auto               # auto-detect the language (e.g. Korean)

Ctrl+C quits.
"""

import argparse
import time

from mumoco.config import WHISPER_LANGUAGE, WHISPER_MODEL
from mumoco.joystick_input import open_joystick
from mumoco.microphone import Microphone, Transcriber, audio_level_db
from mumoco.utils import read_terminal_line


def device_arg(value):
    return int(value) if value.isdigit() else value


def wait_for_toggle(joystick):
    """Joystick START or Enter."""
    while not ((joystick and "START" in joystick.poll()) or read_terminal_line() is not None):
        time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list audio devices and quit")
    ap.add_argument("--device", type=device_arg, default=None, help="input device index or name (default: system default)")
    ap.add_argument("--model", default=WHISPER_MODEL, help=f"Whisper checkpoint (default: {WHISPER_MODEL})")
    ap.add_argument("--language", default=WHISPER_LANGUAGE, help=f"spoken language, or 'auto' (default: {WHISPER_LANGUAGE})")
    args = ap.parse_args()

    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return

    print(f"[INFO] Loading Whisper model '{args.model}' (first run downloads it)...")
    transcriber = Transcriber(args.model, None if args.language == "auto" else args.language)
    print(f"[INFO] Whisper ready on {transcriber.model.device}.")
    mic = Microphone(device=args.device)
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
                print("[WARNING] Almost silent: check the microphone (--list / --device).")

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
