"""
2D WheeledRobot Navigating Tasks
"""
import numpy
import sys
import pygame
from numpy import random
from numpy import cos, sin
from math import acos, asin
from gym import spaces

T_Pi = 2.0 * numpy.pi
H_Pi = 0.5 * numpy.pi

class WheeledRobot(object):
    def __init__(self, 
            wheel_width=0.50,
            wheel_dia=0.10,
            max_wheel_rotation=5.0,
            dt = 0.20
            ):  # Can set goal to test adaptation.
        #distance between two wheels
        self._wheel_width = wheel_width
        self._wheel_dia = wheel_dia
        self._max_wheel_rotation = max_wheel_rotation
        self._dt = dt
        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Box(low=numpy.array([-1.0, -1.0]), 
                high=numpy.array([1.0, 1.0]), dtype=numpy.float32)
        self.reset()

    @property
    def state(self):
        return [self._pos_x, self._pos_y, self._direction]

    def reset(self):
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._direction = 0.0
        return self.state

    def render(self, surface, dist2pixel, screen_scale):
        c_cur = cos(self._direction)
        s_cur = sin(self._direction)
        l_wheel_x = self._pos_x + 0.5 * s_cur * self._wheel_width
        l_wheel_y = self._pos_y - 0.5 * c_cur * self._wheel_width
        r_wheel_x = self._pos_x - 0.5 * s_cur * self._wheel_width
        r_wheel_y = self._pos_y + 0.5 * c_cur * self._wheel_width
        d_w_x = c_cur * 0.5 * self._wheel_dia
        d_w_y = s_cur * 0.5 * self._wheel_dia

        def pos_2_pixel(x, y):
            p_x = (x + 0.5 * screen_scale) * dist2pixel
            p_y = (0.5 * screen_scale - y) * dist2pixel
            return (p_x, p_y)
        
        pygame.draw.line(surface,  pygame.Color("red"), pos_2_pixel(l_wheel_x, l_wheel_y), pos_2_pixel(r_wheel_x, r_wheel_y), width=1)
        pygame.draw.line(surface,  pygame.Color("black"), pos_2_pixel(l_wheel_x - d_w_x, l_wheel_y - d_w_y), pos_2_pixel(l_wheel_x + d_w_x, l_wheel_y + d_w_y), width=3)
        pygame.draw.line(surface,  pygame.Color("black"), pos_2_pixel(r_wheel_x - d_w_x, r_wheel_y - d_w_y), pos_2_pixel(r_wheel_x + d_w_x, r_wheel_y + d_w_y), width=3)

    def step(self, action):
        #valid actions lie in between -1 and 1
        eff_action = numpy.clip(action, -1, 1).astype("float32")
        assert self.action_space.contains(eff_action), "action not valid: %s"%(eff_action)
        l_dist = eff_action[0] * self._dt * self._wheel_dia * self._max_wheel_rotation
        r_dist = eff_action[1] * self._dt * self._wheel_dia * self._max_wheel_rotation
        deta_r = r_dist - l_dist
        avg_dist = 0.5 * (l_dist + r_dist)
        c_cur = cos(self._direction)
        s_cur = sin(self._direction)
        if(abs(deta_r) < 1.0e-6):
            self._pos_x += avg_dist * c_cur
            self._pos_y += avg_dist * s_cur
        else:
            d_theta = deta_r / self._wheel_width
            rot_r = self._wheel_width * avg_dist / deta_r
            c_dtheta_2 = cos(0.5 * d_theta)
            s_dtheta_2 = sin(0.5 * d_theta)
            c_dtheta = c_dtheta_2 ** 2 - s_dtheta_2 ** 2
            s_dtheta = 2.0 * c_dtheta_2 * s_dtheta_2
            if(abs(c_dtheta_2) > 1.0e-6):
                d_dist = rot_r * s_dtheta / c_dtheta_2
            else:
                d_dist = 0.0
            c_mid = c_dtheta_2 * c_cur - s_dtheta_2 * s_cur
            s_mid = c_cur * s_dtheta_2 + s_cur * c_dtheta_2
            self._pos_x += d_dist * c_mid
            self._pos_y += d_dist * s_mid
            self._direction += d_theta
            while self._direction > T_Pi:
                self._direction -= T_Pi
            while self._direction < 0:
                self._direction += T_Pi

        return self.state
