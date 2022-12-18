"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame
from numpy import random

from gym import error, spaces, utils
from gym.utils import seeding

class Bandits(gym.Env):
    def __init__(self, 
            arms = 10,
            max_steps = 5000,
            ):

        self.max_steps = max_steps

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Discrete(arms)

        # observation is the x, y coordinate of the grid
        self.observation_space = None 

        # expected ctr of each arms 
        self.exp_gains = None
        self.K = arms
        self.need_reset = True

        assert self.K > 1 and self.max_steps > 1

    def sample_task(self,
            distribution_settings="Classical",
            mean=0.50,
            dev=0.05,
            ):
        """
        Classical: 1 arm is mean + sqrt(K-1) dev, others are mean - dev / sqrt(K-1)
        Uniform:   each arm with expected gain being Uniform(mean - 1.732 * dev, mean + 1.732 * dev)
        Gaussian:   Gaussian(mean, dev)
        """
        if(distribution_settings == "Classical"):
            fac = numpy.sqrt(self.K - 1)
            exp_gains = numpy.full((self.K,), mean - dev / fac)
            sel_idx = random.randint(0, self.K - 1)
            exp_gains[sel_idx] = mean + fac * dev
            exp_gains = numpy.clip(exp_gains, 0.0, 1.0)
        elif(distribution_settings == "Uniform"):
            exp_gains = numpy.clip((random.random(shape=(self.K,)) - 0.50) * 3.464 + mean, 0.0, 1.0).tolist()
        elif(distribution_settings == "Gaussian"):
            exp_gains = numpy.clip(random.normal(loc=mean, scale=dev), 0.0, 1.0)
        else:
            raise Exception("No such distribution_settings: %s", distribution_settings)
        return exp_gains

    def set_task(self, task_config):
        self.exp_gains = task_config
        assert numpy.shape(self.exp_gains) == (self.K,)
        self.need_reset = True

    def reset(self):
        if(self.exp_gains is None):
            raise Exception("Must call \"set_task\" before reset")
        self.steps = 0
        self.need_reset = False
        return None

    def step(self, action):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        exp_gain = self.exp_gains[action]
        reward = 1 if random.random() < exp_gain else 0
        info = {"steps": self.steps, "expected_gain": exp_gain}
        self.steps += 1
        done = self.steps >= self.max_steps
        if(done):
            self.need_reset = True

        return None, reward, done, info

    def expected_upperbound(self):
        return self.max_steps * numpy.max(self.exp_gains)
