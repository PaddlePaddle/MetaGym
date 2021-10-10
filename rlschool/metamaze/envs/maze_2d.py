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

class MazeCore2D(object):
    def __init__(self):
        pygame.init()

    def set_task(self, task_config, textures):
        # initialize textures
        self._cell_walls = numpy.copy(task_config.cell_walls)
        self._start = task_config.start
        self._max_cells = numpy.shape(self._cell_walls)[0]
        self._cell_walls[task_config.goal[0], task_config.goal[1]] = -1
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def reset(self):
        self._agent_pos = numpy.copy(self._start)
        self.update_observation()
        return self.get_observation()

    def do_action(self, action):
        assert numpy.shape(action) == (2,)
        assert abs(action[0]) < 2 and abs(action[1]) < 2
        tmp_pos_i = self._agent_pos[0] + action[0]
        tmp_pos_j = self._agent_pos[1] + action[1]
        if(self._cell_walls[tmp_pos_i, tmp_pos_j] < 1):
            self._agent_pos[0] = tmp_pos_i
            self._agent_pos[1] = tmp_pos_j
        done = (self._cell_walls[tmp_pos_i, tmp_pos_j] < 0)
        self.update_observation()
        return done

    def render_init(self, view_size, god_view):
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._max_cells
        self._view_size = view_size
        self._god_view = god_view

        self._obs_logo = self._font.render("Observation", 0, pygame.Color("red"))
        if(self._god_view):
            self._screen = pygame.display.set_mode((2 * view_size, view_size))
            pygame.display.set_caption("RandomMazeRender - GodView")
            self._surf_god = pygame.Surface((view_size, view_size))
            self._surf_god.fill(pygame.Color("white"))
            logo_god = self._font.render("GodView", 0, pygame.Color("red"))
            it = numpy.nditer(self._cell_walls, flags=["multi_index"])
            for _ in it:
                x,y = it.multi_index
                if(self._cell_walls[x,y] > 0):
                    pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size, view_size - (y + 1) * self._render_cell_size,
                            self._render_cell_size, self._render_cell_size), width=0)
                if(self._cell_walls[x,y] < 0):
                    pygame.draw.rect(self._surf_god, pygame.Color("green"), (x * self._render_cell_size, view_size - (y + 1) * self._render_cell_size,
                            self._render_cell_size, self._render_cell_size), width=0)
            self._surf_god.blit(logo_god,(view_size - 90, 5))
        else:
            self._screen = pygame.display.set_mode((view_size, view_size))
            pygame.display.set_caption("MetaMazeRender")

    def render_update(self, god_view):
        #Paint Observation
        obs_array = numpy.full((3,3,3), 255, dtype="int32")
        obs_array[numpy.where(self._observation == 1)] = numpy.asarray([0, 0, 0], dtype="int32")
        obs_array[numpy.where(self._observation == -1)] = numpy.asarray([0, 255, 0], dtype="int32")
        obs_array[1,1] = numpy.asarray([255, 0, 0], dtype="int32")
        obs_array = obs_array[:,::-1]
        
        empty_range = 40
        obs_surf = pygame.surfarray.make_surface(obs_array)
        obs_surf = pygame.transform.scale(obs_surf, (self._view_size - 2 * empty_range, self._view_size - 2 * empty_range))
        self._screen.fill(pygame.Color("white"))
        self._screen.blit(self._obs_logo,(5, 5))
        self._screen.blit(obs_surf, (empty_range, empty_range))
        pygame.draw.rect(self._screen, pygame.Color("blue"), 
                (empty_range, empty_range,
                self._view_size - 2 * empty_range, self._view_size - 2 * empty_range), width=1)
        #Paint God View
        if(god_view):
            self._screen.blit(self._surf_god, (self._view_size, 0))
            pygame.draw.rect(self._screen, pygame.Color("red"), 
                    (self._agent_pos[0] * self._render_cell_size + self._view_size, self._view_size - (self._agent_pos[1] + 1) * self._render_cell_size,
                    self._render_cell_size, self._render_cell_size), width=0)

        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()
        return done, keys

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
        self._observation = self._cell_walls[self._agent_pos[0] - 1 : self._agent_pos[0] + 2, self._agent_pos[1] - 1 : self._agent_pos[1] + 2]

    def get_observation(self):
        return numpy.copy(self._observation)
