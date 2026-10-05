"""Video of one participant's corrections of a task: four PyBullet replays on one time axis.

    python3 correction_video.py pnp_spill 0          # -> correction_videos/pnp_spill_user0.mp4

Top left: the answer task. Top right / bottom left / bottom right: the task with the participant's newest
physical-only / language-only / multimodal correction (corrections/<task>/<modality folder>/). All panels
start the task together. A correction panel stops where the participant pressed START and from then on
shows the correction in real time: a colored frame while in correction mode, the recorded motion, and the
spoken words in a speech bubble as they were said. Scene and camera as in sim_task.py.
Needs imageio-ffmpeg (pip install imageio-ffmpeg) for the MP4.
"""

import argparse
import io
import json
import re

import numpy as np
import pybullet
from PIL import Image, ImageDraw, ImageFont

from mumoco.config import CORRECTION_DIR, CORRECTION_FOLDERS, ROOT, VIDEO_FPS, VIDEO_HOLD
from mumoco.sim import find_correction_start, make_simulation, plan_task, play_correction, play_task
from mumoco.utils import find_task_path, load_task, task_base_name

OUT_DIR = ROOT / "correction_videos"
VIDEO_W, VIDEO_H, HEADER_H = 1920, 1080, 40    # [px] video; header with the task, user and time
PANEL_W, PANEL_H, TITLE_H = VIDEO_W // 2, (VIDEO_H - HEADER_H) // 2, 44   # one of the 2x2 panels; title bar on top
SCENE_H = PANEL_H - TITLE_H
PANELS = [("Answer", None), ("Physical Only", "physical"), ("Language Only", "language"), ("Multimodal", "multimodal")]
BORDER = 6                                      # [px] correction-mode frame
INK, MUTED, BAR = (40, 44, 52), (120, 126, 138), (32, 35, 42)
GREEN, BLUE, GRAY, LIGHT = (47, 179, 104), (42, 143, 189), (110, 114, 122), (236, 238, 242)
FONT_FILES = {False: ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
              True: ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]}


def font(size, bold=False):
    for name in FONT_FILES[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


FONTS = {"title": font(23, True), "pill": font(14, True), "header": font(18, True), "label": font(12, True),
         "speech": font(20), "speech_now": font(20, True)}


class JpegFrames(list):
    """Simulation.frames that keeps each frame JPEG-compressed (raw frames of four panels need gigabytes)."""

    def append(self, image):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=93)
        super().append(buffer.getvalue())

    def image(self, k):
        return Image.open(io.BytesIO(self[min(k, len(self) - 1)])).convert("RGB")


# ---------------- correction files ----------------
def find_correction(task_name, user, modality):
    """Newest correction file of the user for the task and modality, or None."""
    name = re.compile(rf"_{re.escape(user)}_{modality}_\d{{8}}_\d{{6}}\.json$")
    paths = [path for path in (CORRECTION_DIR / task_name / CORRECTION_FOLDERS[modality]).glob("*.json")
             if name.search(path.name)]
    return max(paths, key=lambda path: path.name[-20:]) if paths else None      # date_time.json sorts by time


def correction_stop(correction, path):
    """(t, q, gripper_open) where the task stopped for the correction. Files saved before "stop" was added:
    the recording start (multimodal) or the first trajectory sample (physical); language needs "stop"."""
    stop, trajectory = correction.get("stop") or {}, correction.get("trajectory") or []
    if stop.get("q") is None and not trajectory:
        raise SystemExit(f"[ERROR] {path.relative_to(ROOT)} does not store where the robot stopped "
                         "(recorded before 'stop' was saved). Record this correction again.")
    first = trajectory[0] if trajectory else {}
    t = stop.get("t", correction["audio"]["start"] if "audio" in correction else first.get("t"))
    q = stop["q"] if stop.get("q") is not None else first["q"]
    return t, q, stop.get("gripper_open", first.get("gripper_open"))


def correction_events(correction, stop_t, shift):
    """Video times [s] (file time + shift) of the correction: start (START), end (START again), motion span,
    speech segments and words."""
    trajectory = correction.get("trajectory") or []
    segments = (correction.get("language") or {}).get("segments", [])
    ends = [stop_t] + [s["end"] for s in segments] + ([trajectory[-1]["t"]] if trajectory else [])
    if "audio" in correction:
        ends.append(correction["audio"]["start"] + correction["audio"]["duration"])
    return {"start": stop_t + shift, "end": max(ends) + shift,
            "motion": (trajectory[0]["t"] + shift, trajectory[-1]["t"] + shift) if trajectory else None,
            "segments": [(s["start"] + shift, s["end"] + shift) for s in segments],
            "words": [(w["start"] + shift, w["end"] + shift, w["word"]) for s in segments for w in s["words"]]}


# ---------------- simulation ----------------
def simulate(base, waypoints, correction=None, path=None):
    """Frames of one panel (VIDEO_FPS, from video time 0) and its correction events (None: plain task)."""
    sim = make_simulation(base, waypoints, gui=False)
    sim.size = (PANEL_W, SCENE_H)
    sim.record()
    sim.frames = JpegFrames()
    t0 = sim.time
    sim.wait(VIDEO_HOLD[0])                        # start pose
    plan = plan_task(waypoints)
    events = None
    if correction is None:
        play_task(sim, plan)
    else:
        stop_t, q, gripper_open = correction_stop(correction, path)
        segment, t, distance = find_correction_start(plan, q, gripper_open)
        if distance > 0.1:
            print(f"[WARNING] {path.name}: the robot stopped {distance:.2f} rad away from the task path.")
        play_task(sim, plan, stop=(segment, t))
        shift = sim.time - t0 - stop_t             # file time -> video time
        events = correction_events(correction, stop_t, shift)
        if correction.get("trajectory"):
            play_correction(sim, correction, wait=events["motion"][0] - (sim.time - t0))
        sim.wait(max(0.0, events["end"] - (sim.time - t0)))
    sim.wait(VIDEO_HOLD[1])                        # end pose, released object settles
    pybullet.disconnect()
    return sim.frames, events


# ---------------- drawing ----------------
def pill(draw, right, y, text, fill, color=(255, 255, 255)):
    """Rounded label whose right edge is at `right`."""
    w = draw.textlength(text, font=FONTS["pill"])
    box = (right - w - 26, y, right, y + 26)
    draw.rounded_rectangle(box, radius=13, fill=fill)
    draw.text((box[0] + 13, y + 13), text, font=FONTS["pill"], fill=color, anchor="lm")


def status(events, t):
    """Status label and color of a panel at video time t."""
    if events is None:
        return "NO CORRECTION RECORDED", GRAY
    if t < events["start"]:
        return "EXECUTING TASK", GRAY
    if t >= events["end"]:
        return "CORRECTION DONE", (78, 82, 90)
    doing = []
    if events["motion"] and events["motion"][0] <= t <= events["motion"][1]:
        doing.append("GUIDING")
    if any(a <= t <= b for a, b in events["segments"]):
        doing.append("SPEAKING")
    return " · ".join(["CORRECTION MODE"] + [" + ".join(doing)] * bool(doing)), GREEN


def speech_bubble(panel, words, t):
    """Words said up to video time t, in a bubble in the sky at the top right of the scene, clear of the robot
    (the word being said in bold green; long speech scrolls)."""
    said = [(start, end, word) for start, end, word in words if start <= t]
    if not said:
        return
    width, pad, line_h = int(PANEL_W * 0.47), 16, 26
    draw = ImageDraw.Draw(panel)
    lines, line, x = [], [], 0.0                   # wrap word by word
    for start, end, word in said:
        f = FONTS["speech_now"] if start <= t < end else FONTS["speech"]
        w = draw.textlength(word + " ", font=f)
        if line and x + w > width - 2 * pad:
            lines.append(line)
            line, x = [], 0.0
        line.append((word, f))
        x += w
    lines = (lines + [line])[-3:]
    height = pad + 18 + len(lines) * line_h + pad - 6
    right, top = PANEL_W - 14, TITLE_H + 12
    box = (right - width, top, right, top + height)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    o.rounded_rectangle(box, radius=18, fill=(255, 255, 255, 240), outline=(200, 204, 212, 255), width=2)
    tail = [(box[2] - 64, box[3] - 1), (box[2] - 32, box[3] - 1), (box[2] - 14, box[3] + 18)]
    o.polygon(tail, fill=(255, 255, 255, 240))
    o.line([tail[0], tail[2], tail[1]], fill=(200, 204, 212, 255), width=2)
    panel.alpha_composite(overlay)
    draw = ImageDraw.Draw(panel)
    draw.text((box[0] + pad, box[1] + pad - 4), "PARTICIPANT", font=FONTS["label"], fill=MUTED)
    for row, words_in_line in enumerate(lines):
        x, y = box[0] + pad, box[1] + pad + 18 + row * line_h
        for word, f in words_in_line:
            draw.text((x, y), word, font=f, fill=GREEN if f is FONTS["speech_now"] else INK)
            x += draw.textlength(word + " ", font=f)


def draw_panel(scene, title, events, t):
    panel = Image.new("RGBA", (PANEL_W, PANEL_H), BAR)
    panel.paste(scene, (0, TITLE_H))
    draw = ImageDraw.Draw(panel)
    draw.text((18, TITLE_H / 2), title, font=FONTS["title"], fill=(255, 255, 255), anchor="lm")
    if events is None and title == "Answer":
        pill(draw, PANEL_W - 14, 9, "HOW IT SHOULD BE DONE", BLUE)
    else:
        pill(draw, PANEL_W - 14, 9, *status(events, t))
    if events and events["start"] <= t < events["end"]:
        draw.rectangle((BORDER / 2, TITLE_H + BORDER / 2, PANEL_W - BORDER / 2, PANEL_H - BORDER / 2),
                       outline=GREEN, width=BORDER)
    if events:
        speech_bubble(panel, events["words"], t)
    return panel.convert("RGB")


def compose(panels, t, caption):
    frame = Image.new("RGB", (VIDEO_W, VIDEO_H), LIGHT)
    draw = ImageDraw.Draw(frame)
    draw.text((18, HEADER_H / 2), caption, font=FONTS["header"], fill=INK, anchor="lm")
    draw.text((VIDEO_W - 18, HEADER_H / 2), f"t = {t:5.1f} s", font=FONTS["header"], fill=INK, anchor="rm")
    legend = "Correction mode (frame)"                # what the green frame means
    x = VIDEO_W / 2 - draw.textlength(legend, font=FONTS["pill"]) / 2
    draw.rectangle((x - 30, HEADER_H / 2 - 8, x - 14, HEADER_H / 2 + 8), outline=GREEN, width=4)
    draw.text((x, HEADER_H / 2), legend, font=FONTS["pill"], fill=MUTED, anchor="lm")
    for k, (scene, title, events) in enumerate(panels):
        frame.paste(draw_panel(scene, title, events, t), ((k % 2) * PANEL_W, HEADER_H + (k // 2) * PANEL_H))
    draw.line([(PANEL_W, HEADER_H), (PANEL_W, VIDEO_H)], fill=BAR, width=4)
    draw.line([(0, HEADER_H + PANEL_H), (VIDEO_W, HEADER_H + PANEL_H)], fill=BAR, width=4)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Video of a participant's corrections of a task (2x2 replays).")
    parser.add_argument("task", help="task name in tasks/ (without .json)")
    parser.add_argument("user", help="user ID")
    args = parser.parse_args()
    task_path, base = find_task_path(args.task), task_base_name(args.task)
    answer_path = find_task_path(f"{base}_answer")
    if task_path is None or answer_path is None:
        parser.error(f"need tasks/{args.task}.json and task_answers/{base}_answer.json")
    import imageio_ffmpeg                          # fail now, not after the simulations

    panels = []
    for title, modality in PANELS:
        if modality is None:
            print(f"[SIM] {title}: {answer_path.stem}")
            frames, events = simulate(base, load_task(answer_path)["waypoints"])
        else:
            path = find_correction(args.task, args.user, modality)
            print(f"[SIM] {title}: {path.relative_to(ROOT) if path else 'no correction of this user -> task only'}")
            correction = json.loads(path.read_text()) if path else None
            frames, events = simulate(base, load_task(task_path)["waypoints"], correction, path)
        panels.append((frames, title, events))

    count = max(len(frames) for frames, _, _ in panels)
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{args.task}_user{args.user}.mp4"
    writer = imageio_ffmpeg.write_frames(str(out), (VIDEO_W, VIDEO_H), fps=VIDEO_FPS, quality=8,
                                         macro_block_size=8)
    writer.send(None)
    caption = f"{args.task}  ·  user {args.user}"
    for k in range(count):
        t = k / VIDEO_FPS
        frame = compose([(frames.image(k), title, events) for frames, title, events in panels], t, caption)
        writer.send(np.asarray(frame).tobytes())
        if k % (5 * VIDEO_FPS) == 0:
            print(f"[INFO] Writing video {t:5.1f} / {count / VIDEO_FPS:.1f} s", flush=True)
    writer.close()
    print(f"[INFO] {out.relative_to(ROOT)} ({count / VIDEO_FPS:.1f} s)")


if __name__ == "__main__":
    main()
