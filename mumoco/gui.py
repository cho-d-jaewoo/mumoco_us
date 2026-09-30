"""Tkinter window for main.py.

Keyboard, mouse and joystick are all turned into the same actions
(up, down, confirm, yes, no, cancel, correct, open, close); each screen decides what they do.
The experiment runs in a worker thread and changes screens only through the public methods
below (Tk itself is used only in the window thread).
"""

import queue
import signal
import threading
import tkinter as tk
import traceback
from tkinter import font as tkfont, messagebox

from .joystick_input import open_joystick
from .utils import Stopped

KEY_ACTIONS = {"<Up>": "up", "<Down>": "down", "<Left>": "up", "<Right>": "down",
               "<Return>": "confirm", "<KP_Enter>": "confirm", "<Escape>": "cancel",
               "y": "yes", "Y": "yes", "n": "no", "N": "no", "o": "open", "O": "open", "c": "close", "C": "close",
               "<Control-c>": "correct"}
JOY_ACTIONS = {"UP": "up", "DOWN": "down", "Y": "yes", "X": "no", "START": "correct", "A": "open", "B": "close"}

ORANGE, GREEN, BLUE, PURPLE = "#ff9900", "#a0d4a4", "#2a8fbd", "#8d5fd3"
LIGHT_GRAY, DARK_GRAY, WHITE = "#b3b3b3", "#666666", "#ffffff"
BG, FG, MUTED, ACCENT, WARN = WHITE, DARK_GRAY, LIGHT_GRAY, BLUE, PURPLE   # MUTED: borders, disabled items
BANNERS = {"wait": ("PLEASE WAIT", LIGHT_GRAY, DARK_GRAY),       # mode: (text, background, text color)
           "moving": ("ROBOT MOVING", ORANGE, DARK_GRAY),
           "input": ("WAITING FOR INPUT", BLUE, WHITE),
           "correction": ("CORRECTION MODE", GREEN, DARK_GRAY),
           "error": ("ERROR", PURPLE, WHITE)}
FONTS = ("Palatino Linotype", "Palatino", "P052", "TeX Gyre Pagella", "URW Palladio L",
         "DejaVu Sans")                                             # first installed one is used
VISIBLE_ITEMS = 6            # list rows shown at once (the list scrolls with the selection)
POLL_MS = 20


class ExperimentUI:
    def __init__(self, title="Robot Experiment"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1000x720")
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

    def choose(self, title, options, disabled=(), back=False):
        """Blocks until an enabled option is confirmed; returns its index (None = back, if allowed)."""
        self._inbox.put((self._list_screen, (title, options, set(disabled), back)))
        return self._wait_answer()

    def ask_yes_no(self, title, text=""):
        """Blocks until Yes or No is chosen; No is preselected."""
        self._inbox.put((self._yes_no_screen, (title, text)))
        return self._wait_answer()

    def show_execution(self, task_name, request_correction):
        """Task screen; START / Ctrl+C / the button call request_correction() once."""
        self._inbox.put((self._execution_screen, (task_name, request_correction)))

    def show_correction(self):
        """Correction screen; call only once the robot really is in correction mode."""
        while self.pop_commands():
            pass
        self._inbox.put((self._correction_screen, ()))

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
        """Terminal Ctrl+C = START: start/finish a correction while a task runs, otherwise quit."""
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

    def _answer(self, value):
        if self._waiting:
            self._waiting = False
            self._actions = {}
            self._answers.put(value)

    def _show(self, mode, title, text="", hint="", actions=None):
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
    def _highlight(widget, on, disabled=False):
        """Selected: blue with white text. Otherwise white with a light gray border."""
        edge = ACCENT if on else MUTED
        widget.configure(bg=ACCENT if on else BG, fg=WHITE if on else (MUTED if disabled else FG),
                         highlightbackground=edge, highlightcolor=edge)

    def _text(self, text, size=22, color=FG):
        label = tk.Label(self.body, text=text, font=(self.font, size), bg=BG, fg=color, justify="center")
        label.pack(pady=8)
        return label

    def _list_screen(self, title, options, disabled, back):
        sel, top = [0], [0]

        def refresh():
            top[0] = min(max(top[0], sel[0] - VISIBLE_ITEMS + 1), sel[0])
            for row, label in enumerate(rows):
                i = top[0] + row
                on = i == sel[0]
                text = options[i] + ("  (not available)" if i in disabled else "")
                label.configure(text=("▶  " if on else "     ") + text)
                self._highlight(label, on, i in disabled)

        def move(step):
            sel[0] = min(max(sel[0] + step, 0), len(options) - 1)
            self.subtitle.configure(text="")
            refresh()

        def confirm():
            if sel[0] in disabled:
                self.subtitle.configure(text=f"{options[sel[0]]} correction is not implemented yet.", fg=WARN)
            else:
                self._answer(sel[0])

        def click(row):
            sel[0] = top[0] + row
            refresh()

        actions = {"up": lambda: move(-1), "down": lambda: move(1), "confirm": confirm, "yes": confirm}
        hint = "↑ ↓ / D-pad: move     Enter / Y: confirm"
        if back:
            actions["no"] = actions["cancel"] = lambda: self._answer(None)
            hint += "     Esc / X: back"
        self._show("input", title, "", hint, actions)
        rows = []
        for row in range(min(len(options), VISIBLE_ITEMS)):
            label = tk.Label(self.body, font=(self.font, 24), anchor="w", padx=30, pady=12, width=30,
                             highlightthickness=2, cursor="hand2")
            label.pack(pady=5)
            label.bind("<Button-1>", lambda event, r=row: click(r))
            label.bind("<Double-Button-1>", lambda event: confirm())
            rows.append(label)
        self._button("Confirm", "confirm").pack(pady=24)
        refresh()
        self._waiting = True

    def _yes_no_screen(self, title, text):
        sel = [1]                                        # 0 = Yes, 1 = No (default)

        def refresh():
            for i, button in enumerate(buttons):
                self._highlight(button, i == sel[0])

        def toggle():
            sel[0] = 1 - sel[0]
            refresh()

        actions = {"up": toggle, "down": toggle, "confirm": lambda: self._answer(sel[0] == 0),
                   "yes": lambda: self._answer(True), "no": lambda: self._answer(False),
                   "cancel": lambda: self._answer(False)}
        self._show("input", title, text, "← →: move     Enter: confirm     Y: yes     N / X / Esc: no",
                   actions)
        row = tk.Frame(self.body, bg=BG)
        row.pack(pady=30)
        buttons = [self._button("Yes", "yes", row), self._button("No", "no", row)]
        for button in buttons:
            button.pack(side="left", padx=30)
        refresh()
        self._waiting = True

    def _execution_screen(self, task_name, request_correction):
        def correct():
            self._actions = {}
            request_correction()
            info.configure(text="Stopping the task and entering correction mode...", fg=WARN)

        self._show("moving", "Executing Task", task_name, "", {"correct": correct})
        info = self._text("Robot is performing the task.")
        self._text("To correct the robot, press START on the joystick,\nCtrl+C, or the button below.",
                   size=18)
        self._button("Request Correction", "correct").pack(pady=24)

    def _correction_screen(self):
        def finish():
            self._actions = {}
            self._commands.put("finish")
            info.configure(text="Finishing correction...", fg=WARN)

        actions = {"open": lambda: self._commands.put("open"), "close": lambda: self._commands.put("close"),
                   "correct": finish}                    # START again (or Ctrl+C) ends the correction
        self._show("correction", "Physical Correction",
                   "Press the External Activation Switch\nand guide the robot.", "", actions)
        self._text("A / O:  Open Gripper        B / C:  Close Gripper", size=20)
        info = self._text("When you are done, release the switch and press START again.", size=18)
        row = tk.Frame(self.body, bg=BG)
        row.pack(pady=20)
        for text, action in (("Open Gripper", "open"), ("Close Gripper", "close"), ("Finish", "correct")):
            self._button(text, action, row).pack(side="left", padx=12)
