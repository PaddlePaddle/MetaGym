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
    def __init__(self, view_grid=2):
        self.view_grid = view_grid
        pygame.init()

    def set_task(self, task_config, textures):
        # initialize textures
        self._cell_walls = numpy.copy(task_config.cell_walls)
        self._start = task_config.start
        self._max_cells = numpy.shape(self._cell_walls)[0]
        self._cell_walls[task_config.goal[0], task_config.goal[1]] = -1
        self._step_reward = task_config.step_reward
        self._goal_reward = task_config.goal_reward
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def reset(self):
        self._agent_trajectory = []
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
        w,h = self._observation.shape
        c_w = w // 2
        c_h = h // 2
        obs_array = numpy.full((w,h,3), 255, dtype="int32")
        obs_array[numpy.where(self._observation == 1)] = numpy.asarray([0, 0, 0], dtype="int32")
        obs_array[numpy.where(self._observation == -1)] = numpy.asarray([0, 255, 0], dtype="int32")
        obs_array[c_w, c_h] = numpy.asarray([255, 0, 0], dtype="int32")
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

    def render_trajectory(self, file_name):
        traj_screen = pygame.display.set_mode((self._view_size, self._view_size))
        traj_screen.blit(self._surf_god, (0, 0))
        for i in range(len(self._agent_trajectory)-1):
            p = self._agent_trajectory[i]
            n = self._agent_trajectory[i+1]
            p = [(p[0] + 0.5) * self._render_cell_size, self._view_size - (p[1] + 0.5) *  self._render_cell_size]
            n = [(n[0] + 0.5) * self._render_cell_size, self._view_size - (n[1] + 0.5) *  self._render_cell_size]
            pygame.draw.line(traj_screen, pygame.Color("red"), p, n, width=3)
        pygame.image.save(traj_screen, file_name)

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
        self._observation = numpy.ones(shape=(2 * self.view_grid + 1, 2 * self.view_grid + 1), dtype="float32")
        x_s = self._agent_pos[0] - self.view_grid
        x_e = self._agent_pos[0] + self.view_grid + 1
        y_s = self._agent_pos[1] - self.view_grid
        y_e = self._agent_pos[1] + self.view_grid + 1
        i_s = 0
        i_e = 2 * self.view_grid + 1
        j_s = 0
        j_e = 2 * self.view_grid + 1
        if(x_s < 0):
            i_s = -x_s
            x_s = 0
        if(x_e > self._max_cells):
            i_e -= x_e - self._max_cells
            x_e = self._max_cells
        if(y_s < 0):
            j_s = -y_s
            y_s = 0
        if(y_e > self._max_cells):
            j_e -= y_e - self._max_cells
            y_e = self._max_cells
        self._observation[i_s:i_e, j_s:j_e] = self._cell_walls[x_s:x_e, y_s:y_e]
        self._agent_trajectory.append(numpy.copy(self._agent_pos))

    def get_observation(self):
        return numpy.copy(self._observation)
