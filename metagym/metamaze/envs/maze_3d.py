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
from metagym.metamaze.envs.dynamics import PI, PI_2, PI_4, PI2d, vector_move_with_collision
from metagym.metamaze.envs.ray_caster_utils import maze_view

class MazeCore3D(object):
    #Read Configurations
    def __init__(
            self,
            collision_dist=0.20, #collision distance
            max_vision_range=12.0, #agent vision range
            fol_angle = 0.6 * PI, #Field of View
            resolution_horizon = 320, #resolution in horizontal
            resolution_vertical = 320, #resolution in vertical
            with_guidepost = True, # include guide post in observations
        ):
        self._collision_dist = collision_dist
        self._max_vision_range = max_vision_range
        self._fol_angle = fol_angle
        self._resolution_horizon = resolution_horizon
        self._resolution_vertical = resolution_vertical
        self._with_guidepost = with_guidepost
        pygame.init()

    def get_cell_center(self, cell):
        p_x = cell[0] * self._cell_size + 0.5 * self._cell_size
        p_y = cell[1] * self._cell_size + 0.5 * self._cell_size
        return numpy.array([p_x, p_y], dtype="float32")

    def set_task(self, task_config, textures):
        # initialize textures
        self._textures = textures
        self._arrow_size = self._resolution_horizon // 10
        self._arrow_surf = pygame.surfarray.make_surface(self._textures.arrow)
        self._arrow_surf.set_colorkey([0,0,0])
        self._arrow_surf = pygame.transform.scale(self._arrow_surf, (self._arrow_size, self._arrow_size))

        self._cell_walls = task_config.cell_walls
        self._cell_texts = task_config.cell_texts
        self._start = task_config.start
        self._goal = task_config.goal
        self._cell_size = task_config.cell_size
        self._wall_height = task_config.wall_height
        self._agent_height = task_config.agent_height
        self._max_cells = numpy.shape(self._cell_walls)[0]
        self._step_reward = task_config.step_reward
        self._goal_reward = task_config.goal_reward
        assert self._agent_height < self._wall_height and self._agent_height > 0, "the agent height must be > 0 and < wall height"
        assert self._cell_walls.shape == self._cell_texts.shape, "the dimension of walls must be equal to textures"
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def reset(self):
        self._goal_position = self.get_cell_center(self._goal)
        self._agent_pos = self.get_cell_center(self._start)
        self._agent_trajectory = []
        self._agent_ori = 2.0 * random.random() * PI
        self._max_wh = self._max_cells * self._cell_size
        self._cell_transparents = numpy.zeros_like(self._cell_walls, dtype="int32")
        self._cell_transparents[self._goal] = 1
        self.update_observation()
        return self.get_observation()

    def do_action(self, turn_rate, walk_speed, dt=0.10):
        turn_rate = numpy.clip(turn_rate, -1, 1) * PI
        walk_speed = numpy.clip(walk_speed, -1, 1)
        self._agent_ori, self._agent_pos = vector_move_with_collision(
                self._agent_ori, self._agent_pos, turn_rate, walk_speed, dt, 
                self._cell_walls, self._cell_size, self._collision_dist)
        goal_dist = norm(self._agent_pos - self._goal_position)
        done = False
        self.update_observation()
        if(goal_dist < 0.80): #reaching the goal
            done = True
        return done

    def render_init(self, view_size, god_view):
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._max_cells
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion
        self._view_size = view_size

        if(god_view):
            self._screen = pygame.display.set_mode((view_size * 2, view_size))
            pygame.display.set_caption("RandomMazeRender - GodView")
            self._surf_god = pygame.Surface((view_size, view_size))
            self._surf_god.fill(pygame.Color("white"))
            logo = self._font.render("GodView", 0, pygame.Color("red"))
            it = numpy.nditer(self._cell_walls, flags=["multi_index"])
            for _ in it:
                x,y = it.multi_index
                if(self._cell_walls[x,y] > 0):
                    pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size, y * self._render_cell_size,
                            self._render_cell_size, self._render_cell_size), width=0)
            goal_pos = self._goal_position * self._pos_conversion
            pygame.draw.line(self._surf_god, pygame.Color("red"), (goal_pos[0]- 0.30 * self._pos_conversion, goal_pos[1]), (goal_pos[0] + 0.30 * self._pos_conversion, goal_pos[1]), width=2)
            pygame.draw.line(self._surf_god, pygame.Color("red"), (goal_pos[0], goal_pos[1] - 0.30 * self._pos_conversion), (goal_pos[0], goal_pos[1] + 0.30 * self._pos_conversion), width=2)
            self._surf_god.blit(logo,(view_size - 90, 5))
        else:
            self._screen = pygame.display.set_mode((view_size, view_size))
            pygame.display.set_caption("MetaMazeRender")

    def render_update(self, god_view):
        #Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))
        #Paint God View
        if(god_view):
            agent_pos = self._agent_pos * self._pos_conversion
            dx = self._ori_size * numpy.cos(self._agent_ori)
            dy = self._ori_size * numpy.sin(self._agent_ori)
            self._screen.blit(self._surf_god, (self._view_size, 0))
            pygame.draw.circle(self._screen, pygame.Color("green"), (agent_pos[0] + self._view_size, agent_pos[1]), 0.15 * self._pos_conversion)
            pygame.draw.line(self._screen, pygame.Color("green"), (agent_pos[0] + self._view_size, agent_pos[1]), 
                    (agent_pos[0] + self._view_size + dx, agent_pos[1] + dy), width=1)

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
            p = self._agent_trajectory[i] * self._pos_conversion
            n = self._agent_trajectory[i+1] * self._pos_conversion
            pygame.draw.line(traj_screen, pygame.Color("red"), p, n, width=3)
        pygame.image.save(traj_screen, file_name)

    def movement_control(self, keys):
        #Keyboard control cases
        turn_rate = 0.0
        walk_speed = 0.0
        if keys[pygame.K_LEFT]:
            turn_rate = -0.2
        if keys[pygame.K_RIGHT]:
            turn_rate = 0.2
        if keys[pygame.K_UP]:
            walk_speed = 0.5
        if keys[pygame.K_DOWN]:
            walk_speed = -0.5
        return turn_rate, walk_speed

    def update_observation(self):
        #Add the ground first
        #Find Relative Cells
        surf = pygame
        self._observation = maze_view(self._agent_pos, self._agent_ori, self._agent_height, 
                self._cell_walls, self._cell_transparents, self._cell_texts, self._cell_size, self._textures.grounds,
                self._textures.ceil, self._wall_height, 1.0, self._max_vision_range, 0.20, 
                self._fol_angle, self._resolution_horizon, self._resolution_vertical)
        self._obs_surf = pygame.surfarray.make_surface(self.get_observation())
        self._agent_trajectory.append(numpy.copy(self._agent_pos))
        if(self._with_guidepost):
            goal_ori = (self._goal_position - self._agent_pos)
            goal_ori_norm = norm(goal_ori)
            goal_angle = numpy.arcsin(goal_ori[1] / goal_ori_norm)
            if(goal_ori[0] < 0):
                goal_angle = PI - goal_angle
            goal_angle -= self._agent_ori
            blit_pos = 0.75 * self._arrow_size - 0.5 * self._arrow_size * (abs(numpy.sin(goal_angle)) + abs(numpy.cos(goal_angle)))
            self._obs_surf.blit(pygame.transform.rotate(self._arrow_surf,  - goal_angle * PI2d), (blit_pos, blit_pos))
            self._observation = pygame.surfarray.array3d(self._obs_surf)

    def get_observation(self):
        return numpy.copy(self._observation)
