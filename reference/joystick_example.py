import pygame
import numpy as np

DEADBAND = 0.1


class Joystick(object):

    def __init__(self, step_size_l=0.1, step_size_a=(0.15 * np.pi / 4)):
        """
        Buttons mapped for SteelSeries duo joystick
        """
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False
        self.step_size_l = step_size_l
        self.step_size_a = step_size_a

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)

        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        self.X_pressed = self.gamepad.get_button(2)
        self.Y_pressed = self.gamepad.get_button(3)
        self.Back_pressed = self.gamepad.get_button(6)

        Buttons = (self.A_pressed, self.B_pressed, self.X_pressed, self.Y_pressed, self.Back_pressed)
        start = self.gamepad.get_button(7)
        
        z1 = self.gamepad.get_axis(1) # Left stick (up-down) : (-1 to 1)
        z2 = self.gamepad.get_axis(0) # Left stick (left-right) : (-1 to 1)
        z3 = self.gamepad.get_axis(4) # Right stick (up-down) : (-1 to 1)
        z = [z1, z2, z3]

        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return tuple(z), Buttons, start
    
    def getAction(self, z):
        if self.toggle:
            action = (0, 0, 0, self.step_size_a * z[0], self.step_size_a * z[1], -self.step_size_a * z[2])
        else:
            action = (self.step_size_l * z[0], self.step_size_l * z[1],
            -self.step_size_l * z[2], 0, 0, 0)

        return action
    
    def close(self):
        self.gamepad.quit()
        pygame.quit()