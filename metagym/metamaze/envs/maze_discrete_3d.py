"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from pygame import font
from copy import deepcopy
from numpy import random as npyrnd
from numpy.linalg import norm
from metagym.metamaze.envs.dynamics import PI, PI_2, PI_4, PI2d, vector_move_with_collision
from metagym.metamaze.envs.ray_caster_utils import maze_view
from metagym.metamaze.envs.maze_task import MAZE_TASK_MANAGER
from metagym.metamaze.envs.maze_base import MazeBase

class MazeCoreDiscrete3D(MazeBase):
    #Read Configurations
    def __init__(
            self,
            collision_dist=0.20, #collision distance
            max_vision_range=12.0, #agent vision range
            fol_angle = 0.6 * PI, #Field of View
            resolution_horizon = 320, #resolution in horizontal
            resolution_vertical = 320, #resolution in vertical
            max_steps = 5000,
            task_type = "SURVIVAL"
        ):
        super(MazeCoreDiscrete3D, self).__init__(
                collision_dist = collision_dist,
                max_vision_range = max_vision_range,
                fol_angle = fol_angle,
                resolution_horizon = resolution_horizon,
                resolution_vertical = resolution_vertical,
                task_type = task_type,
                max_steps = max_steps
                )

    def reset(self):
        if(self.task_type == "SURVIVAL"):
            #add the life bar
            self._lifebar_start_x = 0.10 * self.resolution_vertical
            self._lifebar_start_y = 0.10 * self.resolution_vertical
            self._lifebar_w = 0.05 * self.resolution_horizon
            self._lifebar_l = 0.80 * self.resolution_vertical
        self._agent_ori_choice = numpy.asarray([0.0, 0.5, 1.0, 1.5], dtype="float32") * PI
        self._agent_ori_index = 0
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]
        return super(MazeCoreDiscrete3D, self).reset()

    def move(self, step):
        tmp_grid = deepcopy(self._agent_grid)
        if(self._agent_ori_index == 0):
            tmp_grid[0] += step
        elif(self._agent_ori_index == 1):
            tmp_grid[1] += step
        elif(self._agent_ori_index == 2):
            tmp_grid[0] -= step
        elif(self._agent_ori_index == 3):
            tmp_grid[1] -= step
        else:
            raise ValueError("Unexpected agent ori index: %d"%self._agent_ori_index)
        if(tmp_grid[0] >=0 and tmp_grid[0] < self._n and
                tmp_grid[1] >= 0  and tmp_grid[1] < self._n and
                self._cell_walls[tmp_grid[0], tmp_grid[1]] == 0):
            self._agent_grid = tmp_grid
            self._agent_loc = self.get_cell_center(self._agent_grid)

    def turn(self, direction):
        self._agent_ori_index += direction
        self._agent_ori_index = self._agent_ori_index % len(self._agent_ori_choice)
        self._agent_ori = self._agent_ori_choice[self._agent_ori_index]

    def do_action(self, action):
        assert numpy.shape(action) == (2,)
        assert abs(action[0]) < 2 and abs(action[1]) < 2
        self.turn(action[0])
        self.move(action[1])
        reward, done = self.evaluation_rule()
        self.update_observation()
        return reward, done

    def render_init(self, view_size):
        super(MazeCoreDiscrete3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        # Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))

        # Paint God-view
        agent_pos = numpy.array(self._agent_loc) * self._pos_conversion
        dx = self._ori_size * numpy.cos(self._agent_ori)
        dy = self._ori_size * numpy.sin(self._agent_ori)
        pygame.draw.circle(self._screen, pygame.Color("green"), (agent_pos[0] + self._view_size, self._view_size - agent_pos[1]), 0.15 * self._pos_conversion)
        pygame.draw.line(self._screen, pygame.Color("green"), (agent_pos[0] + self._view_size, self._view_size - agent_pos[1]), 
                (agent_pos[0] + self._view_size + dx, self._view_size - agent_pos[1] - dy), width=1)

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
        self._observation = maze_view(self._agent_loc, self._agent_ori, self._agent_height, 
                self._cell_walls, self._cell_transparents, self._cell_texts, self._cell_size, MAZE_TASK_MANAGER.grounds,
                MAZE_TASK_MANAGER.ceil, self._wall_height, 1.0, self.max_vision_range, 0.20, 
                self.fol_angle, self.resolution_horizon, self.resolution_vertical)
        if(self.task_type == "SURVIVAL"):
            lifebar_l = self._life / self._max_life * self._lifebar_l
            start_x = int(self._lifebar_start_x)
            start_y = int(self._lifebar_start_y)
            end_x = int(self._lifebar_start_x + lifebar_l)
            end_y = int(self._lifebar_start_y + self._lifebar_w)
            self._observation[start_x:end_x, start_y:end_y, 0] = 255
            self._observation[start_x:end_x, start_y:end_y, 1] = 0
            self._observation[start_x:end_x, start_y:end_y, 2] = 0
        self._obs_surf = pygame.surfarray.make_surface(self._observation)

    def get_observation(self):
        return numpy.copy(self._observation)
