"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding
from maze_core import Textures, sample_task_config, MazeCore3D


class MetaMaze3D(gym.Env):
    def __init__(self, 
            with_guidpost=True,
            enable_render=True,
            render_scale=480,
            render_godview=True,
            resolution=(320, 320),
            max_steps = 1000,
            ):

        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.render_godview = render_godview
        self.maze_core = MazeCore3D(
                with_guidepost=True,
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                )

        self.max_steps = max_steps

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Box(low=numpy.array([-1.0, -1.0]), 
                high=numpy.array([1.0, 1.0]), dtype=numpy.float32)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(resolution[0], resolution[1], 3), dtype=numpy.float32), 
                high=numpy.full((resolution[0], resolution[1], 3), 256, dtype=numpy.float32),
                dtype=numpy.float32)

        self.textures = Textures("img")
        self.need_reset = True
        self.need_set_task = True

    def sample_task(self,
            cell_scale = 15,
            allow_loops = True,
            cell_size = 2.0,
            wall_height = 3.2,
            agent_height = 1.6
            ):
        return sample_task_config(self.textures.n_texts, 
                max_cells=cell_scale, 
                allow_loops=allow_loops,
                cell_size=cell_size,
                wall_height=wall_height,
                agent_height=agent_height)

    def set_task(self, task_config):
        self.maze_core.set_task(task_config, self.textures) 
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        self.steps = 0
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize, self.render_godview)
        self.need_reset = False
        self.keyboard_press = pygame.key.get_pressed()
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        reward = - 0.1
        self.steps += 1
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(20) # 50 FPS
            tr, ws = self.maze_core.movement_control(self.keyboard_press)
        else:
            tr = action[0]
            ws = action[1]
        done = self.maze_core.do_action(tr, ws)

        if(done):
            reward += 200
        elif(self.steps >= self.max_steps or self.key_done):
            done = True
        if(done):
            self.need_reset=True
        info = {"steps": self.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self):
        self.key_done, self.keyboard_press = self.maze_core.render_update(self.render_godview)
