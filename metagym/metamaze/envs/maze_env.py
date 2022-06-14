"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding
from metagym.metamaze.envs.maze_gen import Textures, sample_task_config
from metagym.metamaze.envs.maze_3d import MazeCore3D
from metagym.metamaze.envs.maze_2d import MazeCore2D

class MetaMaze3D(gym.Env):
    def __init__(self, 
            with_guidepost=True,
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
                with_guidepost = with_guidepost,
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
            allow_loops = False,
            cell_size = 2.0,
            wall_height = 3.2,
            agent_height = 1.6,
            step_reward = -0.01,
            goal_reward = None,
            crowd_ratio = 0.0
            ):
        return sample_task_config(self.textures.n_texts, 
                max_cells=cell_scale, 
                allow_loops=allow_loops,
                cell_size=cell_size,
                wall_height=wall_height,
                crowd_ratio=crowd_ratio,
                step_reward = step_reward,
                goal_reward = goal_reward,
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
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        reward = self.maze_core._step_reward
        self.steps += 1
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(20) # 50 FPS
            tr, ws = self.maze_core.movement_control(self.keyboard_press)
        else:
            tr = action[0]
            ws = action[1]
        done = self.maze_core.do_action(tr, ws)

        if(done):
            reward += self.maze_core._goal_reward
        elif(self.steps >= self.max_steps or self.key_done):
            done = True
        if(done):
            self.need_reset=True
        info = {"steps": self.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        self.key_done, self.keyboard_press = self.maze_core.render_update(self.render_godview)

    def save_trajectory(self, file_name):
        self.maze_core.render_trajectory(file_name)

DISCRETE_ACTIONS=[(-1, 0), (1, 0), (0, -1), (0, 1)]
class MetaMaze2D(gym.Env):
    def __init__(self,
            enable_render=True,
            render_scale=480,
            render_godview=True,
            max_steps = 1000,
            view_grid = 2):
        self.enable_render = enable_render
        self.maze_core = MazeCore2D(view_grid=view_grid)
        self.max_steps = max_steps
        self.render_viewsize = render_scale
        self.render_godview = render_godview

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Discrete(4)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,3), dtype=numpy.int32)

        self.need_reset = True
        self.need_set_task = True

    def sample_task(self,
            cell_scale = 15,
            allow_loops = False,
            cell_size = 2.0,
            wall_height = 3.2,
            agent_height = 1.6,
            step_reward = -0.01,
            goal_reward = None,
            crowd_ratio = 0.0
            ):
        return sample_task_config(2,
                max_cells=cell_scale, 
                allow_loops=allow_loops,
                cell_size=cell_size,
                wall_height=wall_height,
                crowd_ratio=crowd_ratio,
                step_reward = step_reward,
                goal_reward = goal_reward,
                agent_height=agent_height)

    def set_task(self, task_config):
        self.maze_core.set_task(task_config, None) 
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        self.steps = 0
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize, self.render_godview)
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")
        reward = self.maze_core._step_reward
        self.steps += 1
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(100) # 10 FPS
            action = self.maze_core.movement_control(self.keyboard_press)
        else:
            action = DISCRETE_ACTIONS[action]
            
        done = self.maze_core.do_action(action)

        if(done):
            reward += self.maze_core._goal_reward
        elif(self.steps >= self.max_steps or self.key_done):
            done = True
        if(done):
            self.need_reset=True
        info = {"steps": self.steps}

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        self.key_done, self.keyboard_press = self.maze_core.render_update(self.render_godview)

    def save_trajectory(self, file_name):
        self.maze_core.render_trajectory(file_name)
