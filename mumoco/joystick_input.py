"""Joystick press events for record_tasks.py and main.py (pygame, as in joystick_example.py).

Run `python3 -m mumoco.joystick_input` (from the repository root) to print the logical names of pressed buttons (checks the mapping).
"""

import os

from .config import JOY_BUTTONS, JOY_HAT, JOY_NAV_AXIS, JOY_NAV_AXIS_X, JOY_NAV_THRESHOLD


class Joystick:
    """poll() returns the logical buttons pressed since the last call ("A", "B", "X", "Y", "BACK",
    "START", "UP", "DOWN", "LEFT", "RIGHT"). Only the press itself counts: holding a button gives one event."""

    def __init__(self):
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")  # our window is not a pygame window
        import pygame
        pygame.init()
        self.pygame = pygame
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.held = set()
        self.poll()                                      # buttons held at startup do not count

    def _held_now(self):
        g = self.gamepad
        held = {name for name, i in JOY_BUTTONS.items() if g.get_button(i)}
        hat_y = g.get_hat(JOY_HAT)[1] if g.get_numhats() > JOY_HAT else 0
        hat_x = g.get_hat(JOY_HAT)[0] if g.get_numhats() > JOY_HAT else 0
        stick, stick_x = g.get_axis(JOY_NAV_AXIS), g.get_axis(JOY_NAV_AXIS_X)   # up / left are negative
        if hat_y > 0 or stick < -JOY_NAV_THRESHOLD:
            held.add("UP")
        if hat_y < 0 or stick > JOY_NAV_THRESHOLD:
            held.add("DOWN")
        if hat_x < 0 or stick_x < -JOY_NAV_THRESHOLD:
            held.add("LEFT")
        if hat_x > 0 or stick_x > JOY_NAV_THRESHOLD:
            held.add("RIGHT")
        return held

    def poll(self):
        self.pygame.event.get()                          # update the joystick state (joystick_example.py)
        held = self._held_now()
        pressed, self.held = held - self.held, held
        return pressed

    def close(self):
        self.gamepad.quit()
        self.pygame.quit()


def open_joystick():
    """The joystick, or None (keyboard only) if pygame or the device is missing."""
    try:
        joystick = Joystick()
    except Exception as e:
        print(f"[WARNING] Joystick not available ({e}). Keyboard only.")
        return None
    print(f"[INFO] Joystick connected: {joystick.gamepad.get_name()}")
    return joystick


if __name__ == "__main__":                               # check the mapping on the real joystick
    import time
    joystick = open_joystick()
    if joystick is not None:
        g = joystick.gamepad
        print(f"[INFO] {g.get_numbuttons()} buttons, {g.get_numhats()} hats, {g.get_numaxes()} axes. "
              "Press buttons (Ctrl+C to quit).")
        raw = None
        try:
            while True:
                pressed = joystick.poll()
                now = ([i for i in range(g.get_numbuttons()) if g.get_button(i)],
                       [g.get_hat(i) for i in range(g.get_numhats())])
                if now != raw:
                    print(f"raw: buttons down {now[0]}, hats {now[1]}   ->  {sorted(pressed) or ''}")
                    raw = now
                time.sleep(0.01)
        except KeyboardInterrupt:
            joystick.close()
