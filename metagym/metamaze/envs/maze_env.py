"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding
from metagym.metamaze.envs.maze_2d import MazeCore2D
from metagym.metamaze.envs.maze_continuous_3d import MazeCoreContinuous3D
from metagym.metamaze.envs.maze_discrete_3d import MazeCoreDiscrete3D

DISCRETE_ACTIONS=[(-1, 0), (1, 0), (0, -1), (0, 1)]

class MetaMazeDiscrete3D(gym.Env):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            resolution=(320, 320),
            max_steps = 5000,
            task_type = "SURVIVAL"
            ):

        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.maze_core = MazeCoreDiscrete3D(
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                max_steps = max_steps,
                task_type = task_type
                )

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Discrete(4)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(resolution[0], resolution[1], 3), dtype=numpy.float32), 
                high=numpy.full((resolution[0], resolution[1], 3), 256, dtype=numpy.float32),
                dtype=numpy.float32)

        self.need_reset = True
        self.need_set_task = True

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        reward = self.maze_core._step_reward

        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(100) # 10 FPS
            action = self.maze_core.movement_control(self.keyboard_press)
        else:
            action = DISCRETE_ACTIONS[action]
            
        reward, done = self.maze_core.do_action(action)
        if(done):
            self.need_reset=True
        info = {"steps": self.maze_core.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        self.key_done, self.keyboard_press = self.maze_core.render_update()

    def save_trajectory(self, file_name):
        self.maze_core.render_trajectory(file_name)

class MetaMazeContinuous3D(gym.Env):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            resolution=(320, 320),
            max_steps = 5000,
            task_type = "SURVIVAL"
            ):

        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.maze_core = MazeCoreContinuous3D(
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                max_steps = max_steps,
                task_type = task_type
                )

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Box(low=numpy.array([-1.0, -1.0]), 
                high=numpy.array([1.0, 1.0]), dtype=numpy.float32)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(resolution[0], resolution[1], 3), dtype=numpy.float32), 
                high=numpy.full((resolution[0], resolution[1], 3), 256, dtype=numpy.float32),
                dtype=numpy.float32)

        self.need_reset = True
        self.need_set_task = True

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        reward = self.maze_core._step_reward
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(20) # 50 FPS
            tr, ws = self.maze_core.movement_control(self.keyboard_press)
        else:
            tr = action[0]
            ws = action[1]
        reward, done = self.maze_core.do_action(tr, ws)

        if(done):
            self.need_reset=True
        info = {"steps": self.maze_core.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        self.key_done, self.keyboard_press = self.maze_core.render_update()

    def save_trajectory(self, file_name):
        self.maze_core.render_trajectory(file_name)

class MetaMaze2D(gym.Env):
    def __init__(self,
            enable_render=True,
            render_scale=480,
            max_steps = 5000,
            task_type = "SURVIVAL",
            view_grid = 2):
        self.enable_render = enable_render
        self.maze_core = MazeCore2D(view_grid=view_grid, max_steps=max_steps, task_type=task_type)
        self.render_viewsize = render_scale

        # Go EAST/WEST/SOUTH/NORTH
        self.action_space = spaces.Discrete(4)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,3), dtype=numpy.int32)

        self.need_reset = True
        self.need_set_task = True

    def set_task(self, task_config):
        self.maze_core.set_task(task_config) 
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(100) # 10 FPS
            action = self.maze_core.movement_control(self.keyboard_press)
        else:
            action = DISCRETE_ACTIONS[action]
            
        reward, done = self.maze_core.do_action(action)

        if(done):
            self.need_reset=True
        info = {"steps": self.maze_core.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        self.key_done, self.keyboard_press = self.maze_core.render_update()

    def save_trajectory(self, file_name, additional=None):
        self.maze_core.render_trajectory(file_name, additional=additional)
