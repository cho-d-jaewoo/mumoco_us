import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import whisper

from camera import OakCamera, WebCamera
from pi0 import PI0
from franka_panda import Franka, WORKSPACE_MAX, WORKSPACE_MIN
from user_study import UserStudyLogger
from utils import Joystick, ObservationBuffer, preprocess_video

# Condition label shown to the participant; this deployment is the PI0 baseline arm
# of the study.
METHOD_LABEL = "method D"
# Standing reminder on the participant screen: HOME is always available, not just when
# the arm has drifted somewhere the policy handles badly.
HOME_HINT = "Press HOME to move the arm back to the neutral position"
WHISPER_MODEL = "turbo"
GRIPPER_OPEN_COMMAND = "c"
GRIPPER_CLOSE_COMMAND = "o"
# Blend applied PER DOF GROUP (translation, orientation) and only to the group the
# operator is actually driving; a group they leave alone stays 100% policy. The
# joystick emits a pure-linear or pure-angular command depending on its LB toggle,
# so driving translation hands orientation entirely to the policy, and vice versa.
JOYSTICK_WEIGHT = 0.9
POLICY_WEIGHT = 0.1

"""
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
port forwarding:
ssh -N -J [user_name]@tinkercliffs2.arc.vt.edu -L 9000:localhost:9000 [user_name]@[node]
"""


def log(message: str) -> None:
    print(message, flush=True)


def print_idle_instructions() -> None:
    log("=" * 68)
    log("IDLE — controller connected; robot joint velocity is held at zero.")
    log("Press joystick X once to START recording a spoken task.")
    log("Press joystick X again to STOP recording and run Whisper.")
    log("Press joystick Y to cancel/end a task and return to IDLE.")
    log("Press joystick HOME to pause and return the arm to its home position.")
    log("Press q on either view window (or Ctrl+C) to shut down the program.")
    log("=" * 68)


class VoiceTaskRecorder:
    """Non-blocking X-to-start/X-to-stop recorder with background transcription."""

    def __init__(self, model, sample_rate: int = 16_000):
        self.model = model
        self.sample_rate = sample_rate
        # Whisper casts weights per-op rather than storing them halved, so fp16 buys
        # speed, not memory: measured on turbo/2070S it cuts a transcription from
        # 0.75 s to 0.38 s at an unchanged 4.9 GB peak. CPU has no fp16 kernels, so
        # inference there stays FP32.
        self.fp16 = getattr(model, "device", None) is not None and model.device.type == "cuda"
        self.state = "idle"
        self.stream = None
        self.audio_chunks = []
        self.audio_status = []
        self.results = queue.Queue()
        self.generation = 0
        self.model_lock = threading.Lock()

    def _audio_callback(self, indata, _frames, _time_info, status):
        if status:
            self.audio_status.append(str(status))
        self.audio_chunks.append(indata.copy())

    def start(self) -> None:
        if self.state != "idle":
            return
        self.audio_chunks = []
        self.audio_status = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        self.state = "recording"
        log("RECORDING — speak the task now.")
        log("Press joystick X again to stop and transcribe.")

    def stop_and_transcribe(self) -> None:
        if self.state != "recording":
            return
        self.stream.stop()
        self.stream.close()
        self.stream = None

        if not self.audio_chunks:
            self.state = "idle"
            log("No audio was captured. Returned to IDLE.")
            print_idle_instructions()
            return

        audio = np.concatenate(self.audio_chunks, axis=0).reshape(-1)
        duration = len(audio) / self.sample_rate
        if self.audio_status:
            log(f"Audio warning(s): {'; '.join(self.audio_status)}")

        self.state = "transcribing"
        generation = self.generation
        log(f"Recording stopped ({duration:.1f} seconds). Transcribing...")

        def worker():
            try:
                with self.model_lock:
                    result = self.model.transcribe(audio, fp16=self.fp16)
                task = result.get("text", "").strip()
                self.results.put((generation, task, None))
            except Exception as exc:
                self.results.put((generation, None, exc))

        threading.Thread(target=worker, daemon=True).start()

    def cancel(self) -> None:
        self.generation += 1
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.audio_chunks = []
        self.audio_status = []
        self.state = "idle"

    def poll(self) -> str | None:
        try:
            generation, task, error = self.results.get_nowait()
        except queue.Empty:
            return None
        if generation != self.generation:
            return None

        self.state = "idle"
        if error is not None:
            log(f"Whisper transcription failed: {error}")
            print_idle_instructions()
            return None
        if not task:
            log("Whisper returned an empty task. Returned to IDLE.")
            print_idle_instructions()
            return None
        log(f"Whisper task: {task!r}")
        return task

    def close(self) -> None:
        self.cancel()


def clear_policy_work(policy: PI0, prompt: str | None = None) -> None:
    """Drop buffered/pending client-side work before switching task state."""
    if prompt is not None:
        policy.prompt = prompt
    with policy._obs_lock:
        policy._latest_obs = None
    policy._obs_event.clear()
    with policy._act_lock:
        policy.action_buffer.clear()


def run_homing(robot: Franka, conn, policy: PI0) -> None:
    """Stop the arm, drive it back to the home joint configuration, then drop the stale
    policy work queued against the pre-homing pose.

    This BLOCKS -- go2position runs its own control loop until it converges or times out
    (20 s) -- and that block is the intended pause: while it runs neither the policy nor
    the joystick can command the arm. Used to recover when the joints drift somewhere the
    policy was never trained on and its actions stop making sense."""
    robot.send2robot(conn, np.zeros(7, dtype=np.float32))
    robot.go2position(conn)
    robot.send2robot(conn, np.zeros(7, dtype=np.float32))
    clear_policy_work(policy)      # buffered actions were computed for the OLD pose


def read_robot_observation(robot: Franka, conn, gripper_state: float):
    state = robot.readState(conn)
    agent_pos = np.concatenate(
        [np.asarray(state["q"], dtype=np.float32), [gripper_state]]
    )
    return state, agent_pos


def get_camera_image(camera) -> np.ndarray:
    """Return 224x224 RGB HWC in the format expected by PI0."""
    bgr, _ = camera.capture()
    return preprocess_video(
        bgr,
        specs={"cvtColor": "BGR2RGB", "resize": (224, 224)},
    )


def make_observation(
    camera: OakCamera, wrist_camera: WebCamera | None, agent_pos: np.ndarray
) -> dict:
    obs = {"image": get_camera_image(camera), "agent_pos": agent_pos}
    if wrist_camera is not None:
        # Consumed by PI0 as observation/wrist_image_left; when the wrist cam is
        # absent the key is omitted and PI0 falls back to a zero image.
        obs["wrist_image"] = get_camera_image(wrist_camera)
    return obs


def human_tag(moving_pos: bool, moving_ang: bool) -> str:
    """Which DOF groups the operator is currently driving, for the live view."""
    if moving_pos and moving_ang:
        return "HUMAN"
    if moving_pos:
        return "HUMAN-LIN"
    if moving_ang:
        return "HUMAN-ANG"
    return "ROBOT"


def at_workspace_limit(robot_state: dict) -> bool:
    """True when the EE sits on a workspace bound, i.e. where clip_xdot_to_workspace
    starts zeroing outward motion. Uses the same >=/<= test as the clip itself, so the
    view flags the boundary on exactly the ticks the clip can engage."""
    ee = np.asarray(robot_state["x"][:3], dtype=np.float64)
    return bool(np.any(ee <= WORKSPACE_MIN) or np.any(ee >= WORKSPACE_MAX))


# BGR, to match OpenCV's channel order
WHITE, GREEN, YELLOW = (255, 255, 255), (140, 235, 140), (80, 220, 255)


def user_prompt(executing: bool, voice_state: str) -> tuple[str, tuple]:
    """The one line of guidance shown to the participant, for the current state."""
    if executing:
        return "Press Y to end this task and start a new one", GREEN
    if voice_state == "recording":
        return "Listening - say your task, then press X again", YELLOW
    if voice_state == "transcribing":
        return "Processing your task...", YELLOW
    return "Press X to speak a new task", WHITE


def quit_pressed(*keys: int | None) -> bool:
    """True if 'q' came back from any view window this tick. cv2.waitKey returns -1
    when no key was pressed, and the key code needs masking to a byte."""
    return any(k is not None and k != -1 and (k & 0xFF) in (ord("q"), ord("Q"))
               for k in keys)


def format_timers(session_t0: float, task_t0: float) -> str:
    """Session clock (since startup) and per-task clock (since the user last entered a
    goal, or since the last task ended), for the debug view's footer."""
    now = time.time()
    return f"t {now - session_t0:5.1f}s | task {now - task_t0:5.1f}s"


def draw_center_text(canvas: np.ndarray, text: str, y: int, scale: float = 1.2,
                     thickness: int = 3, color=(255, 255, 255)) -> int:
    """Draw horizontally centred text, wrapped to the canvas width. Returns the y of
    the last line drawn, so callers can stack blocks."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_w = int(canvas.shape[1] * 0.9)
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, font, scale, thickness)[0][0] > max_w and line:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)

    step = int(cv2.getTextSize("Ag", font, scale, thickness)[0][1] * 1.8)
    for i, text_line in enumerate(lines):
        (w, _), _ = cv2.getTextSize(text_line, font, scale, thickness)
        y_line = y + i * step
        cv2.putText(canvas, text_line, ((canvas.shape[1] - w) // 2, y_line),
                    font, scale, color, thickness, cv2.LINE_AA)
    return y + max(0, len(lines) - 1) * step


class StudyVideoRecorder:
    """Throttled MJPG recorder for the debug view, alongside the user-study pickle.

    The writer is created lazily on the first frame (it needs the canvas size) and
    throttled to `fps`, so the .avi plays back at roughly real time even though the
    control loop renders faster."""

    def __init__(self, path: str, fps: float = 20.0):
        self.path = path
        self.fps = max(1e-3, float(fps))
        self.writer = None
        self.size = None
        self._last = 0.0
        self.frames = 0

    def write(self, canvas: np.ndarray) -> None:
        now = time.time()
        if now - self._last < 1.0 / self.fps:
            return
        self._last = now
        if self.writer is None:
            h, w = canvas.shape[:2]
            self.size = (w, h)
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, self.size)
            if not self.writer.isOpened():
                self.writer = None
                log(f"Could not open video writer at {self.path}; not recording video.")
                self.fps = 1e-9      # effectively stop retrying every tick
                return
            log(f"Debug-view video -> {self.path}")
        # a mid-run size change would be silently dropped by VideoWriter, so force it
        if (canvas.shape[1], canvas.shape[0]) != self.size:
            canvas = cv2.resize(canvas, self.size, interpolation=cv2.INTER_NEAREST)
        self.writer.write(canvas)
        self.frames += 1

    def close(self) -> str | None:
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        return self.path


class UserView:
    """Clean participant-facing screen: the condition label, the task they spoke, and a
    single line telling them which button to press next.

    Point this at the participant and keep ObservationViewer for the operator -- it
    deliberately carries none of the debug overlays (camera panels, blend authority,
    workspace flag, timers)."""

    WINDOW = "user study"

    def __init__(self, size=(1280, 720), enabled: bool = True):
        self.size = (int(size[0]), int(size[1]))
        self.enabled = enabled
        self._created = False

    def show(self, task: str | None, prompt: str = "", prompt_color: tuple = WHITE,
             hint: str = HOME_HINT) -> int | None:
        """Draw the screen; returns the key pressed on this window (cv2.waitKey), so the
        caller can honour q without a global keyboard listener. `hint` is the standing
        HOME reminder below the prompt -- pass "" to drop it (e.g. while homing)."""
        if not self.enabled:
            return None
        w, h = self.size
        canvas = np.zeros((h, w, 3), np.uint8)
        draw_center_text(canvas, METHOD_LABEL, y=70, scale=1.1, thickness=2,
                         color=(150, 150, 150))
        if task:
            draw_center_text(canvas, task, y=h // 2, scale=1.5, thickness=3, color=WHITE)
        if prompt:
            # sits above the hint, which is anchored to the bottom edge
            draw_center_text(canvas, prompt, y=h - 110, scale=1.0, thickness=2,
                             color=prompt_color)
        if hint:
            # dimmer and smaller than the prompt: a standing reminder, not the next step
            draw_center_text(canvas, hint, y=h - 45, scale=0.75, thickness=1,
                             color=(130, 130, 130))
        try:
            if not self._created:
                cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(self.WINDOW, w, h)
                self._created = True
            cv2.imshow(self.WINDOW, canvas)
            return cv2.waitKey(1)
        except cv2.error as exc:
            self.enabled = False
            log(f"User view disabled (no display?): {exc}")
            return None

    def close(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        try:
            cv2.destroyWindow(self.WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass


class ObservationViewer:
    """Operator/debug window showing exactly the 224x224 uint8 frames handed to PI0.

    Renders the post-preprocessing arrays (same BGR2RGB + resize + uint8 cast the
    policy sees), so a wrong colour order or a mis-framed camera shows up here
    rather than as mystery actions. Disables itself after one warning when no
    display is available, so the same command still runs headless over SSH.
    """

    WINDOW = "PI0 observation"
    LABELS = {"image": "exterior_image_1_left", "wrist_image": "wrist_image_left"}

    def __init__(self, scale: int = 2, enabled: bool = True):
        self.scale = max(1, int(scale))
        self.enabled = enabled

    def _panel(self, frame: np.ndarray, label: str) -> np.ndarray:
        # uint8 cast first: matches PI0._convert2le, so clipping/rounding is visible.
        bgr = cv2.cvtColor(np.asarray(frame).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if self.scale > 1:
            # INTER_NEAREST keeps the real 224x224 pixels visible instead of
            # smoothing over the detail the policy actually receives.
            bgr = cv2.resize(bgr, None, fx=self.scale, fy=self.scale,
                             interpolation=cv2.INTER_NEAREST)
        bgr = cv2.copyMakeBorder(bgr, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(bgr, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return bgr

    def render(self, obs: dict, status: str = "", timers: str = "") -> np.ndarray:
        """Build the debug canvas. Display-free, so it still works headless (the
        user-study video is recorded from exactly this image)."""
        panels = [
            self._panel(obs[key], label)
            for key, label in self.LABELS.items()
            if key in obs
        ]
        canvas = np.hstack(panels) if len(panels) > 1 else panels[0]
        if status or timers:
            canvas = cv2.copyMakeBorder(canvas, 0, 22, 0, 0, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            font, baseline = cv2.FONT_HERSHEY_SIMPLEX, canvas.shape[0] - 7
            if status:
                cv2.putText(canvas, status[:60], (6, baseline), font, 0.45,
                            (180, 220, 180), 1, cv2.LINE_AA)
            if timers:                                  # right-aligned, so it never
                (w, _), _ = cv2.getTextSize(timers, font, 0.45, 1)   # collides with status
                cv2.putText(canvas, timers, (canvas.shape[1] - w - 6, baseline), font,
                            0.45, (80, 220, 255), 1, cv2.LINE_AA)
        return canvas

    def show(self, obs: dict, status: str = "", timers: str = "",
             recorder: StudyVideoRecorder | None = None) -> int | None:
        """Render, optionally record, then display. Recording works even with the
        window disabled (--no-view), since only the imshow needs a display. Returns the
        key pressed on this window (cv2.waitKey), or None when nothing was displayed."""
        if not self.enabled and recorder is None:
            return None
        canvas = self.render(obs, status, timers)
        if recorder is not None:
            recorder.write(canvas)
        if not self.enabled:
            return None
        try:
            cv2.imshow(self.WINDOW, canvas)
            return cv2.waitKey(1)
        except cv2.error as exc:
            self.enabled = False
            log(f"Image view disabled (no display?): {exc}")
            return None

    def close(self) -> None:
        if not self.enabled:
            return
        self.enabled = False
        try:
            cv2.destroyWindow(self.WINDOW)
            cv2.waitKey(1)
        except cv2.error:
            pass


def run_controller(
    robot_port: int = 8080,
    gripper_port: int = 8081,
    no_home: bool = False,
    joystick_weight: float = JOYSTICK_WEIGHT,
    policy_weight: float = POLICY_WEIGHT,
    home_button: int | None = None,
    whisper_model_name: str = WHISPER_MODEL,
    audio_sample_rate: int = 16_000,
    wrist_device: int | str = 0,
    no_wrist: bool = False,
    user_study: bool = False,
    study_hz: float = 5.0,
    study_dir: str | None = None,
    study_video_fps: float = 20.0,
    view: bool = True,
    view_scale: int = 2,
) -> None:
    if joystick_weight < 0 or policy_weight < 0:
        raise ValueError("Joystick and policy weights must be non-negative")

    robot = Franka()
    conn = None
    gripper_conn = None
    joystick = None
    camera = None
    wrist_camera = None
    voice = None
    viewer = ObservationViewer(scale=view_scale, enabled=view)
    user_view = UserView(enabled=user_study)
    study_logger = None
    video = None

    try:
        log(f"Loading Whisper model '{whisper_model_name}' (first run downloads it)...")
        whisper_model = whisper.load_model(whisper_model_name)
        log(
            f"Whisper ready: '{whisper_model_name}' on {whisper_model.device}, "
            f"{sum(p.numel() for p in whisper_model.parameters()) / 1e6:.0f}M params."
        )

        log("Connecting to PI0 policy server (no task will execute yet)...")
        policy = PI0(device="cuda", action_scale=14, name="pi0", prompt="")
        log("PI0 policy connection ready.")

        joystick = Joystick(0.05,0.4)
        log(
            f"Joystick ready: {joystick_weight:.2f} joystick / "
            f"{policy_weight:.2f} policy while a stick is deflected, "
            "policy-only when the sticks are centred."
        )

        camera = OakCamera()
        log("Starting OAK-D camera...")
        camera.start()

        if no_wrist:
            log("Wrist webcam disabled — PI0 will receive a zero wrist image.")
        else:
            wrist_camera = WebCamera(device=wrist_device)
            log(f"Starting wrist webcam (device {wrist_device})...")
            wrist_camera.start()

        log(f"Connecting robot :{robot_port} / gripper :{gripper_port} ...")
        conn = robot.connect(robot_port)
        gripper_conn = robot.connect(gripper_port)
        log("Robot and gripper controllers connected.")

        if not no_home:
            log("Homing robot...")
            robot.go2position(conn)
            log("Homing complete.")

        if user_study:
            study_logger = UserStudyLogger(
                study_dir or str(Path(__file__).resolve().parent / "data" / "study"),
                max_hz=study_hz,
            )
            study_logger.start()
            log(f"User-study log -> {study_logger.out_dir} (<= {study_hz:g} Hz)")
            video = StudyVideoRecorder(
                str(Path(study_logger.out_dir) / f"study_{study_logger.session}.avi"),
                fps=study_video_fps,
            )
            log("User-study mode: 'user study' window shows the spoken task only.")

        voice = VoiceTaskRecorder(whisper_model, sample_rate=audio_sample_rate)
        executing = False
        current_task = None
        obs_buffer = None
        gripper_state = 0.0
        previous_buttons = [False] * 5
        # session clock shares t = 0 with the study log; the task clock restarts each
        # time the user enters a new goal, and again when a task ends.
        session_t0 = task_t0 = time.time()
        step_time = 1 / 20
        start_time = 0.0

        print_idle_instructions()
        previous_home = False

        while True:
            curr_time = time.time()
            if curr_time - start_time < step_time:
                time.sleep(0.001)
                continue
            start_time = curr_time

            robot_state, agent_pos = read_robot_observation(robot, conn, gripper_state)
            z, buttons, _start = joystick.getInput()
            # z is deadbanded inside getInput(), so a non-zero command is a
            # deliberate deflection rather than stick drift. Engagement is tracked
            # per DOF group, since getAction() emits linear OR angular, never both.
            human_xdot = np.asarray(joystick.getAction(z), dtype=np.float64)
            human_moving_pos = float(np.linalg.norm(human_xdot[:3])) > 1e-6
            human_moving_ang = float(np.linalg.norm(human_xdot[3:6])) > 1e-6
            human_active = human_moving_pos or human_moving_ang
            buttons = [bool(value) for value in buttons]
            pressed = [
                value and not previous
                for value, previous in zip(buttons, previous_buttons, strict=True)
            ]
            previous_buttons = buttons

            home_now = joystick.home_pressed(home_button)
            home_pressed = home_now and not previous_home
            previous_home = home_now

            if home_pressed:
                # Recovery: takes precedence over every other button this tick.
                log("Joystick HOME pressed — pausing control and homing the robot.")
                # Paint the pause on both screens BEFORE blocking, so a 20 s freeze
                # reads as "homing" rather than as a crash.
                if viewer.enabled or video is not None:
                    viewer.show(
                        make_observation(camera, wrist_camera, agent_pos),
                        status="HOMING — control paused",
                        timers=format_timers(session_t0, task_t0),
                        recorder=video,
                    )
                user_view.show(current_task, "Returning to home position - please wait",
                               YELLOW, hint="")

                run_homing(robot, conn, policy)

                if executing:
                    # The buffered frames/actions describe the pre-homing pose, so the
                    # task restarts its observation history from the homed one.
                    _, agent_pos = read_robot_observation(robot, conn, gripper_state)
                    obs_buffer = ObservationBuffer(
                        buffer_size=4,
                        init_obs=make_observation(camera, wrist_camera, agent_pos),
                    )
                    log(f"Homing complete — resuming task {current_task!r}.")
                else:
                    log("Homing complete — back to IDLE.")
                    print_idle_instructions()
                continue

            open_pressed = pressed[0]    # F310 A
            close_pressed = pressed[1]   # F310 B
            record_pressed = pressed[2]  # F310 X
            end_pressed = pressed[3]     # F310 Y

            if end_pressed:
                if executing:
                    log(f"Joystick Y pressed — ending task {current_task!r}.")
                elif voice.state == "recording":
                    log("Joystick Y pressed — cancelling recording.")
                elif voice.state == "transcribing":
                    log("Joystick Y pressed — cancelling pending transcription.")
                else:
                    log("Joystick Y pressed while already IDLE.")

                executing = False
                current_task = None
                obs_buffer = None
                task_t0 = time.time()          # task ended -> restart the per-task clock
                voice.cancel()
                clear_policy_work(policy, prompt="")
                robot.send2robot(conn, np.zeros(7, dtype=np.float32))
                print_idle_instructions()
                continue

            if record_pressed:
                if executing:
                    log("Task active. Press joystick Y before recording another task.")
                elif voice.state == "idle":
                    try:
                        voice.start()
                    except Exception as exc:
                        voice.cancel()
                        log(f"Could not start microphone recording: {exc}")
                        print_idle_instructions()
                elif voice.state == "recording":
                    voice.stop_and_transcribe()
                else:
                    log("Whisper is still transcribing; joystick X is ignored.")

            new_task = voice.poll()
            if new_task is not None:
                current_task = new_task
                task_t0 = time.time()          # goal entered -> restart the per-task clock
                clear_policy_work(policy, prompt=current_task)
                obs = make_observation(camera, wrist_camera, agent_pos)
                obs_buffer = ObservationBuffer(buffer_size=4, init_obs=obs)
                executing = True
                log(f"EXECUTING task: {current_task!r}")
                log("Press joystick Y to end this task and return to IDLE.")

            if not executing:
                view_key = None
                if viewer.enabled or video is not None:
                    view_key = viewer.show(
                        make_observation(camera, wrist_camera, agent_pos),
                        status=f"IDLE ({voice.state})",
                        timers=format_timers(session_t0, task_t0),
                        recorder=video,
                    )
                prompt, prompt_color = user_prompt(False, voice.state)
                if quit_pressed(view_key, user_view.show(None, prompt, prompt_color)):
                    log("q pressed — shutting down.")
                    break
                if study_logger is not None:
                    study_logger.record(
                        ee_pos=robot_state["x"][:3], ee_euler=robot_state["x"][3:6],
                        buttons=buttons, human_xdot=human_xdot,
                        robot_xdot=np.zeros(6), task=None,
                    )
                robot.send2robot(conn, np.zeros(7, dtype=np.float32))
                continue

            if open_pressed and close_pressed:
                log("Ignoring simultaneous gripper open/close buttons.")
            elif open_pressed:
                robot.send2gripper(gripper_conn, GRIPPER_OPEN_COMMAND)
                gripper_state = 0.0
                log("Joystick gripper command: OPEN")
            elif close_pressed:
                robot.send2gripper(gripper_conn, GRIPPER_CLOSE_COMMAND)
                gripper_state = 1.0
                log("Joystick gripper command: CLOSE")

            obs = make_observation(camera, wrist_camera, agent_pos)
            obs_buffer.add(obs)
            view_key = viewer.show(
                obs,
                status=(
                    f"[{human_tag(human_moving_pos, human_moving_ang)}"
                    f"{' LIMIT' if at_workspace_limit(robot_state) else ''}] "
                    f"{current_task}"
                ),
                timers=format_timers(session_t0, task_t0),
                recorder=video,
            )
            prompt, prompt_color = user_prompt(True, voice.state)
            if quit_pressed(view_key, user_view.show(current_task, prompt, prompt_color)):
                log("q pressed — shutting down.")
                break
            action = np.asarray(policy.get_action(obs_buffer.get_buffer()), dtype=np.float64)
            if action.shape != (8,) or not np.all(np.isfinite(action)):
                raise RuntimeError(f"Invalid policy action: {action}")

            policy_qdot = action[:7]
            # Blending is per DOF group, which is a Cartesian notion, so the policy's
            # joint command is mapped through the Jacobian to be blended there.
            policy_xdot = np.asarray(robot_state["J"], dtype=np.float64) @ policy_qdot

            blended = np.empty(6)
            if human_moving_pos:
                blended[:3] = (
                    joystick_weight * human_xdot[:3] + policy_weight * policy_xdot[:3]
                )
            else:
                blended[:3] = policy_xdot[:3]
            if human_moving_ang:
                blended[3:6] = (
                    joystick_weight * human_xdot[3:6] + policy_weight * policy_xdot[3:6]
                )
            else:
                blended[3:6] = policy_xdot[3:6]

            # Workspace box: zero any linear component that would push the EE further
            # outside it. Applied to the BLEND, so it bounds the policy too -- until
            # now only the joystick was clipped and the policy could drive the arm
            # into the low-level libfranka failsafe.
            clipped_xdot = robot.clip_xdot_to_workspace(blended, robot_state["x"][:3])

            # Apply the blended/clipped Cartesian velocity as a CORRECTION to the
            # policy's joint command instead of re-solving from scratch. The EE then
            # moves at exactly clipped_xdot either way, but this keeps the policy's
            # null-space (elbow posture) motion, which a plain xdot2qdot(clipped_xdot)
            # would discard -- measured at ~30% of a typical command on this 7-DOF arm.
            # Sticks centred and nothing clipped -> the correction is zero and the
            # policy drives exactly as it did before.
            command_qdot = policy_qdot + robot.xdot2qdot(
                clipped_xdot - policy_xdot, robot_state
            )
            if not np.all(np.isfinite(command_qdot)):
                raise RuntimeError(f"Invalid blended joint velocity: {command_qdot}")

            mode = "angular" if joystick.toggle else "linear"
            # print(
            #     f"Joystick raw={joystick.raw_axes}, deadbanded={z}, mode={mode}\n"
            #     f"Human xdot={human_xdot} (pos={human_moving_pos} ang={human_moving_ang})\n"
            #     f"Policy xdot={policy_xdot}\n"
            #     f"Blended xdot={blended} -> clipped={clipped_xdot}\n"
            #     f"Command qdot={command_qdot}",
            #     flush=True,
            # )
            if study_logger is not None:
                study_logger.record(
                    ee_pos=robot_state["x"][:3], ee_euler=robot_state["x"][3:6],
                    buttons=buttons, human_xdot=human_xdot,
                    robot_xdot=policy_xdot, task=current_task,
                )
            robot.send2robot(conn, command_qdot)

    except KeyboardInterrupt:
        log("Ctrl+C received — shutting down.")
    finally:
        if video is not None:                   # finalize the debug-view recording
            video_path = video.close()
            log(f"Debug-view video saved: {video_path} ({video.frames} frames)"
                if video_path else "Debug-view video recorded no frames.")
        user_view.close()
        if study_logger is not None:            # flush the session to disk
            study_path = study_logger.close()
            log(f"User-study data saved: {study_path}" if study_path
                else "User-study logger recorded no samples.")
        viewer.close()
        if voice is not None:
            voice.close()
        if conn is not None:
            try:
                robot.send2robot(conn, np.zeros(7, dtype=np.float32))
            except OSError:
                pass
        if camera is not None:
            camera.stop()
        if wrist_camera is not None:
            wrist_camera.stop()
        if joystick is not None:
            joystick.close()
        if gripper_conn is not None:
            gripper_conn.close()
        if conn is not None:
            conn.close()
        log("Controller shut down.")


def wrist_device_arg(value: str) -> int | str:
    """Accept either a V4L2 index ('0') or a device path ('/dev/video0')."""
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Persistent PI0/OAK-D voice controller")
    ap.add_argument("--robot-port", type=int, default=8080)
    ap.add_argument("--gripper-port", type=int, default=8081)
    ap.add_argument("--no-home", action="store_true")
    ap.add_argument(
        "--home-button",
        type=int,
        default=None,
        help="Joystick button index that pauses control and homes the arm "
             "(default: 8, the F310's Logitech 'home' button)",
    )
    ap.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=(
            "Whisper checkpoint: turbo (default), medium.en for the largest "
            "English-only model, small.en for lower latency. large-v3 is more "
            "accurate but OOMs on an 8 GB GPU"
        ),
    )
    ap.add_argument("--audio-sample-rate", type=int, default=16_000)
    ap.add_argument(
        "--joystick-weight",
        type=float,
        default=JOYSTICK_WEIGHT,
        help="Joystick share while the operator is deflecting a stick",
    )
    ap.add_argument(
        "--policy-weight",
        type=float,
        default=POLICY_WEIGHT,
        help="Policy share while the operator is deflecting a stick; the policy "
             "always gets the full command when the sticks are centred",
    )
    ap.add_argument(
        "--wrist-device",
        type=wrist_device_arg,
        default=0,
        help="V4L2 index or path of the webcam on top of the OAK-D (default: 0)",
    )
    ap.add_argument(
        "--no-wrist",
        action="store_true",
        help="Run without the wrist webcam; PI0 gets a zero wrist image",
    )
    ap.add_argument(
        "--user-study",
        action="store_true",
        help="User-study mode: open a second, clean 'user study' window showing ONLY "
             "the spoken task (point it at the participant; keep the PI0 observation "
             "window for the operator); RECORD a throttled data log (EE pose, "
             "human/policy EE velocity commands, buttons, task); and RECORD the debug "
             "view to video",
    )
    ap.add_argument(
        "--study-hz",
        type=float,
        default=5.0,
        help="Max user-study logging frequency (Hz); the collector is throttled to this",
    )
    ap.add_argument(
        "--study-dir",
        default="/home/collab/semantic_shared_autonomy_user_study_data/saps",
        help="Output dir for user-study recordings (default: baseline/data/study/)",
    )
    ap.add_argument(
        "--study-video-fps",
        type=float,
        default=20.0,
        help="Frame rate of the recorded debug-view video (--user-study); the writer "
             "is throttled to this so playback is ~real time",
    )
    ap.add_argument(
        "--no-view",
        action="store_true",
        help="Do not open the live window of the 224x224 frames sent to PI0",
    )
    ap.add_argument(
        "--view-scale",
        type=int,
        default=2,
        help="Integer upscale for the live observation window (default: 2)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_controller(
        robot_port=args.robot_port,
        gripper_port=args.gripper_port,
        no_home=args.no_home,
        joystick_weight=args.joystick_weight,
        policy_weight=args.policy_weight,
        home_button=args.home_button,
        whisper_model_name=args.whisper_model,
        audio_sample_rate=args.audio_sample_rate,
        wrist_device=args.wrist_device,
        no_wrist=args.no_wrist,
        view=not args.no_view,
        view_scale=args.view_scale,
        user_study=args.user_study,
        study_hz=args.study_hz,
        study_dir=args.study_dir,
        study_video_fps=args.study_video_fps,
    )


if __name__ == "__main__":
    main()

