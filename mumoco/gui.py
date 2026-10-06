"""Tkinter window for main.py.

Keyboard, mouse and joystick are all turned into the same actions
(up, down, left, right, confirm, yes, no, cancel, correct, open, close); each screen decides what they do.
The experiment runs in a worker thread and changes screens only through the public methods
below (Tk itself is used only in the window thread).
"""

import faulthandler
import os
import queue
import signal
import threading
import tkinter as tk
import traceback
from tkinter import font as tkfont, messagebox

from .joystick_input import open_joystick
from .utils import Stopped, task_video_path
from .video import SyncedVideos

KEY_ACTIONS = {"<Up>": "up", "<Down>": "down", "<Left>": "left", "<Right>": "right",
               "<Return>": "confirm", "<KP_Enter>": "confirm", "<Escape>": "cancel",
               "y": "yes", "Y": "yes", "n": "no", "N": "no", "o": "open", "O": "open", "c": "close", "C": "close",
               "<Control-c>": "correct"}
JOY_ACTIONS = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right", "Y": "yes", "X": "no",
               "START": "correct", "A": "open", "B": "close"}

ORANGE, GREEN, BLUE, PURPLE = "#ff9900", "#a0d4a4", "#2a8fbd", "#8d5fd3"
LIGHT_GRAY, DARK_GRAY, WHITE = "#b3b3b3", "#666666", "#ffffff"
BG, FG, MUTED, ACCENT, WARN = WHITE, DARK_GRAY, LIGHT_GRAY, BLUE, PURPLE   # MUTED: borders, completed items
BANNERS = {"wait": ("PLEASE WAIT", LIGHT_GRAY, DARK_GRAY),       # mode: (text, background, text color)
           "moving": ("ROBOT MOVING", ORANGE, DARK_GRAY),
           "input": ("WAITING FOR INPUT", BLUE, WHITE),
           "correction": ("CORRECTION MODE", GREEN, DARK_GRAY),
           "error": ("ERROR", PURPLE, WHITE)}
FONTS = ("Palatino Linotype", "Palatino", "P052", "TeX Gyre Pagella", "URW Palladio L",
         "DejaVu Sans")                                             # first installed one is used
CORRECTION_SCREENS = {"physical": ("Physical Correction", "Press the External Activation Switch\nand guide the robot."),
                      "language": ("Language Correction", "Tell the robot what it should do\nby speaking into the microphone."),
                      "multimodal": ("Multimodal Correction", "Press the External Activation Switch, guide the robot\n"
                                                              "and tell it what it should do through the microphone.")}
VISIBLE_ITEMS = 6            # list rows shown at once (the list scrolls with the selection)
POLL_MS = 20
SHUTDOWN_HINT_MS = 5000      # the force-quit hint appears if shutting down takes longer


class ExperimentUI:
    def __init__(self, title="Robot Experiment"):
        self.root = tk.Tk()
        self.root.title(title)
        width = min(1280, self.root.winfo_screenwidth() - 40)   # large enough for two videos side by side
        height = min(800, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg=BG)
        installed = set(tkfont.families(self.root))
        self.font = next((f for f in FONTS if f in installed), "TkDefaultFont")
        print(f"[INFO] GUI font: {self.font}")
        self.closing = False                 # read by the worker (Robot.should_stop)
        self._inbox = queue.Queue()          # screen changes requested by the worker
        self._answers = queue.Queue()        # answers to choose() / ask_yes_no()
        self._commands = queue.Queue()       # correction commands for the worker: open / close / finish
        self._actions = {}                   # action -> handler of the current screen
        self._waiting = False                # a choose()/ask_yes_no() answer is expected
        self._cleanup = None                 # stops what the current screen runs (video playback)
        self._worker = None
        self.joystick = open_joystick()

        self.banner = tk.Label(self.root, font=(self.font, 18, "bold"), pady=10)
        self.banner.pack(fill="x")
        self.title = tk.Label(self.root, font=(self.font, 34, "bold"), bg=BG, fg=FG, pady=24)
        self.title.pack()
        self.subtitle = tk.Label(self.root, font=(self.font, 22), bg=BG, fg=FG, wraplength=900)
        self.subtitle.pack()
        self.hint = tk.Label(self.root, font=(self.font, 15), bg=BG, fg=FG, pady=18, wraplength=950)
        self.hint.pack(side="bottom", fill="x")
        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(expand=True)

        for key, action in KEY_ACTIONS.items():
            self.root.bind(key, lambda event, a=action: self._on_action(a))
        self.root.protocol("WM_DELETE_WINDOW", self.request_quit)
        signal.signal(signal.SIGINT, lambda *_: self._inbox.put((self._on_sigint, ())))  # terminal Ctrl+C

    # ---------------- called from the worker thread ----------------
    def run(self, experiment):
        """Run experiment(ui) in a worker thread and the window in this (main) thread."""
        def work():
            try:
                experiment(self)
            except Stopped:
                pass
            except Exception as e:
                traceback.print_exc()
                self.error(f"{type(e).__name__}: {e}")
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self.root.after(POLL_MS, self._poll)
        self.root.focus_force()
        self.root.mainloop()
        if self.joystick:
            self.joystick.close()

    def status(self, title, text="", mode="wait"):
        """Information screen without input; mode: wait / moving."""
        self._inbox.put((self._show, (mode, title, text)))

    def error(self, text):
        self._inbox.put((self._show, ("error", "Something Went Wrong", text,
                                      "The robot was stopped. Close this window and restart the program.")))

    def choose(self, title, options, back=False, subtitle="", completed=()):
        """Blocks until an option is confirmed; returns its index (None = back, if allowed).
        completed: shown as completed and skipped (cannot be selected). subtitle: shown under the title."""
        self._inbox.put((self._list_screen, (title, options, back, subtitle, set(completed))))
        return self._wait_answer()

    def choose_scenario(self, title, subtitle, letters, modalities, unavailable=(), completed=(), example=None):
        """Error scenario (letters) and correction modality on one screen. Blocks; returns
        (letter index, modality index), or None to go back. unavailable: letters shown but not selectable;
        completed: (letter index, modality index) pairs already done (a letter with all modalities done is
        completed too). example: a task name whose video "Task Example" shows."""
        self._inbox.put((self._scenario_screen, (title, subtitle, letters, modalities, set(unavailable),
                                                 set(completed), example)))
        return self._wait_answer()

    def ask_user_id(self, title="Enter Your User ID"):
        """Blocks until a numeric user ID is entered; returns it as a string."""
        self._inbox.put((self._user_id_screen, (title,)))
        return self._wait_answer()

    def ask_yes_no(self, title, text=""):
        """Blocks until Yes or No is chosen; No is preselected."""
        self._inbox.put((self._yes_no_screen, (title, text)))
        return self._wait_answer()

    def wait_for_start(self, title, text):
        """Instruction screen; blocks until START (or Ctrl+C) is pressed."""
        self._inbox.put((self._start_screen, (title, text)))
        self._wait_answer()

    def show_execution(self, task_name, request_correction):
        """Task screen; START / Ctrl+C / the button call request_correction() once."""
        self._inbox.put((self._execution_screen, (task_name, request_correction)))

    def show_correction(self, modality="physical"):
        """Correction screen (modality: physical / language / multimodal); call only once the robot really
        is in correction mode (language: stopped, microphone recording)."""
        while self.pop_commands():
            pass
        self._inbox.put((self._correction_screen, (modality,)))

    def pop_commands(self):
        """Correction commands entered since the last call (never blocks)."""
        commands = []
        while not self._commands.empty():
            commands.append(self._commands.get_nowait())
        return commands

    def _wait_answer(self):
        while True:
            try:
                return self._answers.get(timeout=0.1)
            except queue.Empty:
                if self.closing:
                    raise Stopped

    # ---------------- window thread ----------------
    def _poll(self):
        try:
            while not self._inbox.empty():
                fn, args = self._inbox.get_nowait()
                fn(*args)
            if self.joystick:
                for button in self.joystick.poll():
                    if button in JOY_ACTIONS:
                        self._on_action(JOY_ACTIONS[button])
        finally:                                         # an error in one handler must not freeze the window
            if self.closing and not self._worker.is_alive():
                self.root.destroy()
            else:
                self.root.after(POLL_MS, self._poll)

    def _on_action(self, action):
        handler = self._actions.get(action)
        if handler:
            handler()

    def _on_sigint(self):
        """Terminal Ctrl+C = START: start/finish a correction while a task runs, otherwise quit.
        While shutting down: force quit, after printing where the worker hangs."""
        if self.closing:
            print("\n[WARNING] Shutdown did not finish. Where each thread is:", flush=True)
            faulthandler.dump_traceback(all_threads=True)
            os._exit(1)
        if "correct" in self._actions:
            self._on_action("correct")
        else:
            self.request_quit(ask=False)

    def request_quit(self, ask=True):
        if self.closing:
            return
        if ask and self._worker.is_alive() and not messagebox.askyesno(
                "Quit", "Stop the experiment and close the program?"):
            return
        self.closing = True
        self._show("wait", "Shutting Down", "Stopping the robot safely...")
        self.root.after(SHUTDOWN_HINT_MS, self._shutdown_hint)

    def _shutdown_hint(self):
        if self._worker.is_alive():
            print("[WARNING] Still shutting down. Press Ctrl+C in the terminal to force quit.", flush=True)
            self.hint.configure(text="Taking too long? Press Ctrl+C in the terminal to force quit.")

    def _answer(self, value):
        if self._waiting:
            self._waiting = False
            self._actions = {}
            self._answers.put(value)

    def _show(self, mode, title, text="", hint="", actions=None):
        if self._cleanup:
            self._cleanup()
            self._cleanup = None
        self.hint.unbind("<Button-1>")
        self.hint.configure(cursor="")
        banner, color, text_color = BANNERS[mode]
        self.banner.configure(text=banner, bg=color, fg=text_color)
        self.title.configure(text=title)
        self.subtitle.configure(text=text, fg=FG)
        self.hint.configure(text=hint)
        for widget in self.body.winfo_children():
            widget.destroy()
        self._actions = actions or {}

    def _button(self, text, action, parent=None):
        """Large clickable label; a click is the same as the keyboard/joystick action."""
        button = tk.Label(parent or self.body, text=text, font=(self.font, 22, "bold"), padx=40, pady=16,
                          highlightthickness=2, cursor="hand2")
        self._highlight(button, False)
        button.bind("<Button-1>", lambda event: self._on_action(action))
        return button

    @staticmethod
    def _highlight(widget, on, muted=False):
        """Selected: blue with white text. Otherwise white with a light gray border."""
        edge = ACCENT if on else MUTED
        widget.configure(bg=ACCENT if on else BG, fg=WHITE if on else (MUTED if muted else FG),
                         highlightbackground=edge, highlightcolor=edge)

    def _text(self, text, size=22, color=FG):
        label = tk.Label(self.body, text=text, font=(self.font, size), bg=BG, fg=color, justify="center")
        label.pack(pady=8)
        return label

    def _list_screen(self, title, options, back, subtitle="", completed=()):
        """Highlighted list; Up / Down move (completed items are skipped), Y confirms."""
        selectable = [i for i in range(len(options)) if i not in completed]
        state = {"sel": selectable[0] if selectable else 0, "top": 0}

        def refresh():
            state["top"] = min(max(state["top"], state["sel"] - VISIBLE_ITEMS + 1), state["sel"])
            for row, label in enumerate(rows):
                i = state["top"] + row
                on = i == state["sel"]
                text = options[i] + ("  [Completed]" if i in completed else "")
                label.configure(text=("\u25B6  " if on else "     ") + text)
                self._highlight(label, on, i in completed)

        def move(step):
            ahead = [i for i in selectable if (i - state["sel"]) * step > 0]
            if ahead:
                state["sel"] = min(ahead, key=lambda i: abs(i - state["sel"]))
                refresh()

        def confirm():
            if state["sel"] not in completed:
                self._answer(state["sel"])

        def click(row):
            if state["top"] + row not in completed:
                state["sel"] = state["top"] + row
                refresh()

        actions = {"up": lambda: move(-1), "down": lambda: move(1), "confirm": confirm, "yes": confirm}
        if back:
            actions["no"] = actions["cancel"] = lambda: self._answer(None)
        self._show("input", title, subtitle, "\u2191 / \u2193 : Move     Y : Confirm", actions)
        rows = []
        for row in range(min(len(options), VISIBLE_ITEMS)):
            label = tk.Label(self.body, font=(self.font, 24), anchor="w", padx=30, pady=12, width=30,
                             highlightthickness=2, cursor="hand2")
            label.pack(pady=5)
            label.bind("<Button-1>", lambda event, r=row: click(r))
            label.bind("<Double-Button-1>", lambda event: confirm())
            rows.append(label)
        refresh()
        self._waiting = True

    def _scenario_screen(self, title, subtitle, letters, modalities, unavailable, completed, example, state=None):
        """Three columns: "Task Example" | error scenarios | correction modalities (shown once a scenario is
        chosen). Left / Right move between the columns, Up / Down within one, Y confirms, X goes back."""
        done = {i for i in range(len(letters)) if all((i, m) in completed for m in range(len(modalities)))}
        open_letters = [i for i in range(len(letters)) if i not in unavailable and i not in done]
        st = state or {"focus": "letters", "letter": open_letters[0] if open_letters else 0, "chosen": None,
                       "modality": 0}

        def open_modalities():
            return [m for m in range(len(modalities)) if st["chosen"] is not None and (st["chosen"], m) not in completed]

        def refresh():
            for i, label in enumerate(letter_rows):
                on = i == st["letter"]
                tag = "  [Completed]" if i in done else "  (not available)" if i in unavailable else ""
                label.configure(text=("\u25B6  " if on else "     ") + letters[i] + tag)
                self._highlight(label, on and st["focus"] == "letters", i in done or i in unavailable)
                if i == st["chosen"] and st["focus"] != "letters":    # the chosen scenario stays marked
                    label.configure(highlightbackground=ACCENT, highlightcolor=ACCENT)
            for m, label in enumerate(modality_rows):
                if st["chosen"] is None:                               # modalities appear once a scenario is chosen
                    label.configure(text="", bg=BG, highlightbackground=BG, highlightcolor=BG)
                    continue
                on = m == st["modality"] and st["focus"] == "modalities"
                finished = (st["chosen"], m) in completed
                label.configure(text=("\u25B6  " if on else "     ") + modalities[m] + ("  [Completed]" if finished else ""))
                self._highlight(label, on, finished)
            self._highlight(example_button, st["focus"] == "example")

        def move(step):
            column = {"letters": open_letters, "modalities": open_modalities()}.get(st["focus"], [])
            key = "letter" if st["focus"] == "letters" else "modality"
            ahead = [i for i in column if (i - st[key]) * step > 0]
            if ahead:
                st[key] = min(ahead, key=lambda i: abs(i - st[key]))
                refresh()

        def choose_letter(i):
            if i in open_letters:
                st["letter"], st["chosen"], st["focus"] = i, i, "modalities"
                st["modality"] = (open_modalities() or [0])[0]
                refresh()

        def confirm():
            if st["focus"] == "example":
                show_example()
            elif st["focus"] == "letters":
                choose_letter(st["letter"])
            elif st["modality"] in open_modalities():
                self._answer((st["chosen"], st["modality"]))

        def left():
            st["focus"] = {"modalities": "letters", "letters": "example"}.get(st["focus"], st["focus"])
            refresh()

        def right():
            st["focus"] = {"example": "letters", "letters": "modalities" if st["chosen"] is not None else "letters"
                           }.get(st["focus"], st["focus"])
            refresh()

        def back():
            if st["focus"] == "modalities":                            # undo the scenario choice
                st["focus"], st["chosen"] = "letters", None
                refresh()
            else:
                self._answer(None)

        def show_example():
            self._example_screen(subtitle, example, lambda: self._scenario_screen(
                title, subtitle, letters, modalities, unavailable, completed, example, st))

        def click_modality(m):
            if m in open_modalities():
                st["modality"], st["focus"] = m, "modalities"
                refresh()

        actions = {"up": lambda: move(-1), "down": lambda: move(1), "left": left, "right": right,
                   "confirm": confirm, "yes": confirm, "no": back, "cancel": back, "example": show_example}
        self._show("input", title, subtitle,
                   "\u2191 / \u2193 : Move     \u2190 / \u2192 : Switch Column     Y : Confirm     X : Back", actions)
        columns = tk.Frame(self.body, bg=BG)
        columns.pack()
        example_button = self._button("Task Example", "example", columns)
        example_button.pack(side="left", padx=(0, 50))
        letter_column, modality_column = tk.Frame(columns, bg=BG), tk.Frame(columns, bg=BG)
        letter_column.pack(side="left", padx=(0, 40))
        modality_column.pack(side="left")
        letter_rows, modality_rows = [], []
        for i in range(len(letters)):
            label = tk.Label(letter_column, font=(self.font, 24), anchor="w", padx=24, pady=10, width=14,
                             highlightthickness=2, cursor="hand2")
            label.pack(pady=5)
            label.bind("<Button-1>", lambda event, i=i: choose_letter(i))
            letter_rows.append(label)
        for m in range(len(modalities)):
            label = tk.Label(modality_column, font=(self.font, 24), anchor="w", padx=24, pady=10, width=20,
                             highlightthickness=2, cursor="hand2")
            label.pack(pady=5)
            label.bind("<Button-1>", lambda event, m=m: click_modality(m))
            label.bind("<Double-Button-1>", lambda event: confirm())
            modality_rows.append(label)
        refresh()
        self._waiting = True

    def _example_screen(self, task_title, example, back):
        """The answer task's video: how the robot should behave (the goal of a correction)."""
        self._show("input", "Correction Goal", f"{task_title}: how the robot should behave", "X : Back",
                   {action: back for action in ("confirm", "yes", "no", "cancel")})
        self.hint.bind("<Button-1>", lambda event: self._on_action("no"))    # mouse: click "X : Back"
        self.hint.configure(cursor="hand2")
        self.root.update_idletasks()
        width = min(self.root.winfo_width() - 200, int((self.root.winfo_height() - 330) * 16 / 9))
        size = (max(width, 320), max(width, 320) * 9 // 16)
        box = tk.Frame(self.body, width=size[0], height=size[1], bg=BG, highlightthickness=2, highlightbackground=MUTED)
        box.pack_propagate(False)
        box.pack()
        video = tk.Label(box, bg=BG, fg=FG, font=(self.font, 18), wraplength=size[0] - 40)
        video.pack(fill="both", expand=True)
        path = task_video_path(example) if example else None
        if path is None or not path.exists():
            video.configure(text=f"Task example video not found for: {example}")
            return
        try:
            player = SyncedVideos(self.root, [video], [path], size,
                                  on_error=lambda i, message: video.configure(text=message, image=""))
        except ImportError:
            video.configure(text="Video playback needs Pillow (pip install pillow).")
        else:
            self._cleanup = player.stop

    def _yes_no_screen(self, title, text):
        sel = [1]                                        # 0 = Yes, 1 = No (default)

        def refresh():
            for i, button in enumerate(buttons):
                self._highlight(button, i == sel[0])

        def toggle():
            sel[0] = 1 - sel[0]
            refresh()

        actions = {"up": toggle, "down": toggle, "left": toggle, "right": toggle,
                   "confirm": lambda: self._answer(sel[0] == 0), "yes": lambda: self._answer(sel[0] == 0),
                   "pick_yes": lambda: self._answer(True), "pick_no": lambda: self._answer(False)}   # mouse clicks
        self._show("input", title, text, "Y : Confirm", actions)
        row = tk.Frame(self.body, bg=BG)
        row.pack(pady=40)
        buttons = [self._button("Yes", "pick_yes", row), self._button("No", "pick_no", row)]
        for button in buttons:
            button.configure(font=(self.font, 32, "bold"), width=6, pady=36)
            button.pack(side="left", padx=40)
        refresh()
        self._waiting = True

    def _user_id_screen(self, title):
        def submit():
            user_id = entry.get().strip()
            if user_id.isdigit():
                self._answer(user_id)
            else:
                message.configure(text="Please enter a numeric User ID.")

        self._show("input", title, "", "Enter : Continue", {"confirm": submit})
        entry = tk.Entry(self.body, font=(self.font, 26), width=20, justify="center", fg=FG, relief="flat",
                         highlightthickness=2, highlightbackground=MUTED, highlightcolor=ACCENT)
        entry.pack(pady=20, ipady=8)
        entry.focus_set()
        self._button("Continue", "confirm").pack(pady=16)
        message = tk.Label(self.body, font=(self.font, 18), bg=BG, fg=WARN)
        message.pack()
        self._waiting = True

    def _start_screen(self, title, text):
        self._show("input", title, "", "START : Begin", {"correct": lambda: self._answer(True)})
        self._text(text)
        self._waiting = True

    def _execution_screen(self, task_name, request_correction):
        def correct():
            self._actions = {}
            request_correction()
            info.configure(text="Stopping the task and entering correction mode...", fg=WARN)

        self._show("moving", "Executing Task", task_name, "", {"correct": correct})
        info = self._text("Robot is performing the task.")
        self._text("To correct the robot, press START.",
                   size=18)

    def _correction_screen(self, modality):
        def finish():
            self._actions = {}
            self._commands.put("finish")
            info.configure(text="Finishing correction...", fg=WARN)

        title, text = CORRECTION_SCREENS[modality]
        actions = {"correct": finish}                    # START again (or Ctrl+C) ends the correction
        if modality != "language":                       # the gripper is part of the physical correction
            actions.update(open=lambda: self._commands.put("open"), close=lambda: self._commands.put("close"))
        self._show("correction", title, text, "", actions)
        if modality != "physical":
            self._text("\u25cf  Microphone is recording", size=20, color=WARN)
        if modality != "language":
            self._text("A :  Open Gripper        B :  Close Gripper", size=20)
        info = self._text("When you are done, press START again.", size=18)
