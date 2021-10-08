"""
2D WheeledRobot Navigating Tasks
"""
import numpy
import sys
import gym
import pygame
from numpy import random
from numpy import cos, sin
from math import acos, asin
from gym import error, spaces, utils

T_Pi = 6.2831852

# Singal Transmission Decay
def signal_transmission(d, P_o, d_o, n, sigma):
    signal = max(0.0, P_o - n * numpy.log(d / d_o)) + sigma * numpy.abs(random.normal(0.0,1.0)) 
    return numpy.asarray([signal], dtype="float32")

class Navigator2D(gym.Env):
    def __init__(self, robot_class, max_steps, signal_noise, enable_render):  # Can set goal to test adaptation.
        self.max_steps = max_steps
        self.robot = robot_class()
        self.action_space = self.robot.action_space
        self.signal_noise = signal_noise
        self.enable_render = enable_render
        self.observation_space = spaces.Box(low=numpy.array([0.0]), high=numpy.array([numpy.inf]), dtype=numpy.float32)

        self.need_reset = True
        self.need_set_task = True

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        self.steps = 0
        self.need_reset = False
        if(self.enable_render):
            pygame.init()
            resolution = 640
            self.screen = pygame.display.set_mode((resolution, resolution))
            pygame.display.set_caption("Navigator Render")
            self.screen_trajectory = pygame.Surface((resolution, resolution))
            self.screen_trajectory.fill(pygame.Color("white"))
            self.length_pixels = resolution / 10.0
            pygame.draw.circle(self.screen_trajectory, pygame.Color("green"), 
                    self.pos_2_pixel(self.goal[0], self.goal[1]), 0.10 * self.length_pixels)
            self.robot.render(self.screen_trajectory, self.length_pixels, 10.0)
            self.screen.blit(self.screen_trajectory, (0, 0))
            pygame.display.update()
        return self.observation

    def pos_2_pixel(self, x, y):
        p_x = (x + 5.0) * self.length_pixels
        p_y = (5.0 - y) * self.length_pixels
        return (p_x, p_y)

    def sample_task(self):
        return {"Goal_Position": random.uniform(-5, 5, size=(2,)),
                "Signal_Strength_A": random.uniform(20.0, 50.0),
                "Signal_Decay_k": random.uniform(5.0, 10.0)}

    def set_task(self, task):
        self.goal = task["Goal_Position"]
        self.sig_A = task["Signal_Strength_A"]
        self.sig_k = task["Signal_Decay_k"]
        self.need_set_task = False

    @property
    def observation(self):
        dist = numpy.linalg.norm(self.robot.state[:2] - self.goal)
        return signal_transmission(dist, self.sig_A, 0.10, self.sig_k, self.signal_noise)

    def step(self, action):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        self.steps += 1

        #valid actions lie in between -1 and 1
        eff_action = numpy.clip(action, -1, 1)
        self.robot.step(eff_action)
        dist = numpy.linalg.norm(self.robot.state[:2] - self.goal)
        done = (self.steps >= self.max_steps or dist < 0.10)
        reward = - dist

        # Notice that the reward can not be used as instant observations
        return self.observation, reward, done, {"steps": self.steps, "robot_state": self.robot.state}

    def render(self):
        self.robot.render(self.screen_trajectory, self.length_pixels, 10.0)
        self.screen.blit(self.screen_trajectory, (0, 0))
        pygame.display.update()
        pygame.time.delay(50)
        for event in pygame.event.get():
            pass

if __name__=="__main__":
    # running random policies
    from wheeled_robot import WheeledRobot
    game = Navigator2D(WheeledRobot, 200, 0.20, True)
    game.set_task(game.sample_task())
    game.reset()
    done = False
    while not done:
        action = game.action_space.sample()
        action[0] += 0.5
        action[1] += 0.5
        obs, r, done, info = game.step(action)
        game.render()
        print(obs, info)
