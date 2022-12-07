"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from metagym.metamaze.envs.maze_base import MazeBase

class MazeCore2D(MazeBase):
    def __init__(self, view_grid=2, task_type="SURVIVAL", max_steps=5000):
        super(MazeCore2D, self).__init__(
                view_grid=view_grid,
                task_type=task_type,
                max_steps=max_steps
                )

    def do_action(self, action):
        assert numpy.shape(action) == (2,)
        assert abs(action[0]) < 2 and abs(action[1]) < 2
        tmp_grid_i = self._agent_grid[0] + action[0]
        tmp_grid_j = self._agent_grid[1] + action[1]

        if(self._cell_walls[tmp_grid_i, tmp_grid_j] < 1):
            self._agent_grid[0] = tmp_grid_i
            self._agent_grid[1] = tmp_grid_j
        self._agent_loc = self.get_cell_center(self._agent_grid)

        reward, done = self.evaluation_rule()
        self.update_observation()
        return reward, done

    def visualize_obs(self):
        w,h = self._observation.shape
        c_w = w // 2
        c_h = h // 2
        obs_array = numpy.full((w,h,3), 255, dtype="int32")
        obs_array[numpy.where(self._observation < -0.50)] = numpy.asarray([0, 0, 0], dtype="int32")
        for idxes in numpy.argwhere(self._observation > 0.01):
            tidx = tuple(idxes)
            if(tidx==(c_w, c_h)):
                continue
            if(self.task_type == "SURVIVAL"):
                f = int(255 - 255 * self._observation[tidx])
            elif(self.task_type == "ESCAPE"):
                f = 0
            else:
                f = 255
            obs_array[tidx] = numpy.asarray([f, 255, f], dtype="int32")
        if(self.task_type == "SURVIVAL"):
            f = max(0, int(255 - 128 * self._life))
            obs_array[c_w, c_h] = numpy.asarray([255, f, f], dtype="int32")
        else:
            obs_array[c_w, c_h] = numpy.asarray([255, 0, 0], dtype="int32")

        obs_array = obs_array[:,::-1]
        return obs_array

    def render_observation(self):
        #Paint Observation
        empty_range = 40
        obs_surf = pygame.surfarray.make_surface(self.visualize_obs())
        obs_surf = pygame.transform.scale(obs_surf, (self._view_size - 2 * empty_range, self._view_size - 2 * empty_range))
        self._screen.blit(self._obs_logo,(5, 5))
        self._screen.blit(obs_surf, (empty_range, empty_range))

        pygame.draw.rect(self._screen, pygame.Color("blue"), 
                (empty_range, empty_range,
                self._view_size - 2 * empty_range, self._view_size - 2 * empty_range), width=1)
        pygame.draw.rect(self._screen, pygame.Color("red"), 
                (self._agent_grid[0] * self._render_cell_size + self._view_size, self._view_size - (self._agent_grid[1] + 1) * self._render_cell_size,
                self._render_cell_size, self._render_cell_size), width=0)

    def movement_control(self, keys):
        #Keyboard control cases
        if keys[pygame.K_LEFT]:
            return (-1, 0)
        if keys[pygame.K_RIGHT]:
            return (1, 0)
        if keys[pygame.K_UP]:
            return (0, 1)
        if keys[pygame.K_DOWN]:
            return (0, -1)
        return (0, 0)

    def update_observation(self):
        #Add the ground first
        #Find Relative Cells
        self._observation = - numpy.ones(shape=(2 * self.view_grid + 1, 2 * self.view_grid + 1), dtype="float32")
        x_s = self._agent_grid[0] - self.view_grid
        x_e = self._agent_grid[0] + self.view_grid + 1
        y_s = self._agent_grid[1] - self.view_grid
        y_e = self._agent_grid[1] + self.view_grid + 1
        i_s = 0
        i_e = 2 * self.view_grid + 1
        j_s = 0
        j_e = 2 * self.view_grid + 1
        if(x_s < 0):
            i_s = -x_s
            x_s = 0
        if(x_e > self._n):
            i_e -= x_e - self._n
            x_e = self._n
        if(y_s < 0):
            j_s = -y_s
            y_s = 0
        if(y_e > self._n):
            j_e -= y_e - self._n
            y_e = self._n
        # Observation: -1 for walls, 1 for goals, > 0 for foods
        self._observation[i_s:i_e, j_s:j_e] = - self._cell_walls[x_s:x_e, y_s:y_e]
        cur_goal = numpy.zeros((self._n, self._n))
        cur_goal[tuple(self._goal)] = 1.0
        if(self.task_type == "SURVIVAL"):
            self._observation[i_s:i_e, j_s:j_e] += self._cur_food_rewards[x_s:x_e, y_s:y_e]
            self._observation[self.view_grid, self.view_grid] = self._life
        elif(self.task_type == "ESCAPE"):
            self._observation[i_s:i_e, j_s:j_e] += cur_goal[x_s:x_e, y_s:y_e]
