"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
import time
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm

class MazeBase(object):
    def __init__(self, **kw_args):
        for k in kw_args:
            self.__dict__[k] = kw_args[k]
        pygame.init()

    def set_task(self, task_config):
        # initialize textures
        self._cell_walls = numpy.copy(task_config.cell_walls)
        self._cell_texts = task_config.cell_texts
        self._start = task_config.start
        self._n = numpy.shape(self._cell_walls)[0]
        self._goal = task_config.goal
        self._cell_size = task_config.cell_size
        self._wall_height = task_config.wall_height
        self._agent_height = task_config.agent_height
        self._step_reward = task_config.step_reward
        self._goal_reward = task_config.goal_reward
        self._food_rewards = task_config.food_rewards
        self._food_interval = task_config.food_interval
        self._max_life = task_config.max_life
        self._initial_life = task_config.initial_life

        assert self._agent_height < self._wall_height and self._agent_height > 0, "the agent height must be > 0 and < wall height"
        assert self._cell_walls.shape == self._cell_texts.shape, "the dimension of walls must be equal to textures"
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def reset(self):
        self._agent_grid = numpy.copy(self._start)
        self._agent_loc = self.get_cell_center(self._start)
        self._goal_loc = self.get_cell_center(self._goal)
        self._agent_trajectory = [numpy.copy(self._agent_grid)]

        # Maximum w and h in the space
        self._size = self._n * self._cell_size

        # Valid in 3D
        self._agent_ori = 0.0

        if(self.task_type == "SURVIVAL"):
            self._food_wait_refresh = numpy.zeros_like(self._food_rewards, dtype="int32")
            self._cur_food_rewards = numpy.copy(self._food_rewards)
            self._food_revival_count = numpy.copy(self._food_interval)
            self._life = self._initial_life
            self._cell_transparents = self._cur_food_rewards
        elif(self.task_type == "ESCAPE"):
            self._cell_transparents = numpy.zeros_like(self._cell_walls, dtype="int32")
            self._cell_transparents[self._goal] = 1.0
        self.update_observation()
        self.steps = 0
        return self.get_observation()

    def evaluation_rule(self):
        self.steps += 1
        self._agent_trajectory.append(numpy.copy(self._agent_grid))
        agent_grid_idx = tuple(self._agent_grid)

        if(self.task_type == "SURVIVAL"):
            if(self._cur_food_rewards[agent_grid_idx] > 1.0e-2):
                # Get the food
                reward = self._cur_food_rewards[agent_grid_idx]
                self._food_wait_refresh[agent_grid_idx] = 1
                self._cur_food_rewards[agent_grid_idx] = 0.0
            else:
                reward = 0.0
            self._life += reward + self._step_reward
            self._life = min(self._life, self._max_life)
            done = self._life < 0.0 or self.episode_is_over()

            # Refresh the food where necessary
            self._food_revival_count -= self._food_wait_refresh
            for idxes in numpy.argwhere(self._food_revival_count < 0):
                tidx = tuple(idxes)
                self._cur_food_rewards[tidx] = self._food_rewards[tidx]
                self._food_revival_count[tidx] = self._food_interval[tidx]
                self._food_wait_refresh[tidx] = 0

        elif(self.task_type == "ESCAPE"):
            goal = (tuple(self._goal) == agent_grid_idx)
            reward = self._step_reward + goal * self._goal_reward
            done = goal or self.episode_is_over() 

        return reward, done

    def do_action(self, action):
        raise NotImplementedError()

    def render_init(self, view_size):
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._n
        self._view_size = view_size

        self._obs_logo = self._font.render("Observation", 0, pygame.Color("red"))

        self._screen = pygame.Surface((2 * view_size, view_size))
        self._screen = pygame.display.set_mode((2 * view_size, view_size))
        pygame.display.set_caption("RandomMazeRender - GodView")
        self._surf_god = pygame.Surface((view_size, view_size))
        self._surf_god.fill(pygame.Color("white"))
        it = numpy.nditer(self._cell_walls, flags=["multi_index"])
        for _ in it:
            x,y = it.multi_index
            if(self._cell_walls[x,y] > 0):
                pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size, view_size - (y + 1) * self._render_cell_size,
                        self._render_cell_size, self._render_cell_size), width=0)
            if(self.task_type == "ESCAPE" and x == self._goal[0] and y == self._goal[1]):
                pygame.draw.rect(self._surf_god, pygame.Color("green"), (x * self._render_cell_size, view_size - (y + 1) * self._render_cell_size,
                        self._render_cell_size, self._render_cell_size), width=0)
        logo_god = self._font.render("GodView", 0, pygame.Color("red"))
        self._surf_god.blit(logo_god,(view_size - 90, 5))

    def draw_food(self, scr, offset):
        it = numpy.nditer(self._cell_walls, flags=["multi_index"])
        for _ in it:
            x,y = it.multi_index
            if(self._cur_food_rewards[x,y] > 1.0e-2):
                f = int(255 - 255 * self._cur_food_rewards[x,y])
                pygame.draw.rect(scr,  pygame.Color(f, 255, f),
                        (x * self._render_cell_size + offset[0], offset[1] + self._view_size - (y + 1) * self._render_cell_size,
                        self._render_cell_size, self._render_cell_size), width=0)
                txt_life = self._font.render("Life: %f"%self._life, 0, pygame.Color("red"))
                scr.blit(txt_life,(self._view_size + 90, 5))

    def render_observation(self):
        raise NotImplementedError()

    def render_update(self):
        #Paint God View
        self._screen.blit(self._surf_god, (self._view_size, 0))
        if(self.task_type == "SURVIVAL"):
            self.draw_food(self._screen, (self._view_size, 0))

        #Paint Agent and Observation
        self.render_observation()

        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()
        return done, keys

    def render_trajectory(self, file_name, additional=None):
        # Render god view with record on the trajectory
        if(additional is not None):
            aw, ah = additional["surfaces"][0].get_width(),additional["surfaces"][0].get_height()
        else:
            aw, ah = (0, 0)

        traj_screen = pygame.Surface((self._view_size + aw, max(self._view_size, ah)))
        traj_screen.fill(pygame.Color("white"))
        traj_screen.blit(self._surf_god, (0, 0))

        pygame.draw.rect(traj_screen, pygame.Color("red"), 
                (self._agent_grid[0] * self._render_cell_size, self._view_size - (self._agent_grid[1] + 1) * self._render_cell_size,
                self._render_cell_size, self._render_cell_size), width=0)
        if(self.task_type == "SURVIVAL"):
            self.draw_food(traj_screen, (0, 0))

        for i in range(len(self._agent_trajectory)-1):
            p = self._agent_trajectory[i]
            n = self._agent_trajectory[i+1]
            p = [(p[0] + 0.5) * self._render_cell_size, self._view_size - (p[1] + 0.5) *  self._render_cell_size]
            n = [(n[0] + 0.5) * self._render_cell_size, self._view_size - (n[1] + 0.5) *  self._render_cell_size]
            pygame.draw.line(traj_screen, pygame.Color("red"), p, n, width=3)

        # paint some additional surfaces where necessary
        if(additional != None):
            for i in range(len(additional["surfaces"])):
                traj_screen.blit(additional["surfaces"][i], (self._view_size, 0))
                pygame.image.save(traj_screen, file_name.split(".")[0] + additional["file_names"][i] + ".png")
        else:
            pygame.image.save(traj_screen, file_name)

    def episode_is_over(self):
        return self.steps > self.max_steps-1

    def get_cell_center(self, cell):
        p_x = cell[0] * self._cell_size + 0.5 * self._cell_size
        p_y = cell[1] * self._cell_size + 0.5 * self._cell_size
        return [p_x, p_y]

    def get_loc_grid(self, loc):
        p_x = int(loc[0] / self._cell_size)
        p_y = int(loc[1] / self._cell_size)
        return [p_x, p_y]

    def movement_control(self, keys):
        raise NotImplementedError()

    def update_observation(self):
        raise NotImplementedError()

    def get_observation(self):
        return numpy.copy(self._observation)
