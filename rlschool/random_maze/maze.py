import os
import numpy
import pygame
import random
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from dynamics import PI, PI_2, PI_4, PI2d, vector_move_with_collision
from ray_caster_utils import maze_view

# A dict for cell value and corresponding wall

class MazeEnv(object):
    #Read Configurations
    def __init__(
            self,
            cell_size=2.0, #Size of each cell
            wall_height=3.2, #Height of the wall & the ceilings
            collision_dist=0.20, #collision distance
            max_vision_range=12.0, #agent vision range
            fol_angle = 0.6 * PI, #Field of View
            resolution_horizon = 320, #resolution in horizontal
            resolution_vertical = 320, #resolution in vertical
            agent_height = 1.6, # height of agent
            texture_dir = "img", #loading textures
            with_guidepost = True, # guide post
        ):
        self._cell_size = cell_size
        self._wall_height = wall_height
        self._collision_dist = collision_dist
        self._max_vision_range = max_vision_range
        self._fol_angle = fol_angle
        self._resolution_horizon = resolution_horizon
        self._resolution_vertical = resolution_vertical
        self._texture_list = []
        texture_files = os.listdir(texture_dir)
        texture_files.sort()
        for file_name in texture_files:
            if(file_name.find("wall") >= 0):
                self._texture_list.append(pygame.surfarray.array3d(pygame.image.load('/'.join([texture_dir, file_name]))))
            if(file_name.find("ground") >= 0):
                texture_ground = pygame.surfarray.array3d(pygame.image.load('/'.join([texture_dir, file_name])))
            if(file_name.find("ceil") >= 0):
                self._texture_ceil = pygame.surfarray.array3d(pygame.image.load('/'.join([texture_dir, file_name])))
            if(file_name.find("arrow") >= 0):
                self._texture_arrow = pygame.surfarray.array3d(pygame.image.load('/'.join([texture_dir, file_name])))
        self._texture_list = [texture_ground] + self._texture_list
        self._texture_list = numpy.array(self._texture_list)
        self._agent_height = agent_height # agent height is also the vision height
        pygame.init()
        self._arrow_size = resolution_horizon // 10
        self._arrow_surf = pygame.surfarray.make_surface(self._texture_arrow)
        self._arrow_surf.set_colorkey([0,0,0])
        self._arrow_surf = pygame.transform.scale(self._arrow_surf, (self._arrow_size, self._arrow_size))
        self._with_guidepost = with_guidepost
        assert self._agent_height < self._wall_height and self._agent_height > 0, "the agent height must be > 0 and < wall height"

    def is_cell_valid(self, x,y):
        return (x > 0  and x < self._max_cells and y > 0 and y < self._max_cells)

    def is_position_valid(self, pos):
        return (pos[0] > 0 and pos[0] < self._max_wh and pos[1] > 0 and pos[1] < self._max_wh )

    def get_cell_center(self, cell):
        p_x = cell[0] * self._cell_size + 0.5 * self._cell_size
        p_y = cell[1] * self._cell_size + 0.5 * self._cell_size
        return numpy.array([p_x, p_y], dtype="float32")

    def gen_random_maze(self, max_cells=11, allow_loops=True):
        # Initialize the maze ...
        assert max_cells > 6, "Minimum required cells are 7"
        assert max_cells % 2 != 0, "Cell Numbers can only be odd"
        self._max_cells = max_cells
        print("Initializing A Maze...")
        self._cell_walls = numpy.ones(shape=(self._max_cells, self._max_cells), dtype="int32")
        self._cell_texts = numpy.random.randint(1, self._texture_list.shape[0], size=(self._max_cells, self._max_cells))
        # Dig the initial positions
        for i in range(1, self._max_cells, 2):
            for j in range(1, self._max_cells, 2):
                self._cell_walls[i,j] = 0
        #Randomize a start, around the bottom-left
        start_range = self._max_cells / 5
        while True:
            s_x = random.randint(0, (self._max_cells - 1) // 2 - 1) * 2 + 1
            s_y = random.randint(0, (self._max_cells - 1) // 2 - 1) * 2 + 1
            if(s_x <= start_range or s_y <= start_range):
                break
        self._start = (s_x, s_y)

        #Randomize a goal (not quite efficiently)
        self._goal = (self._max_cells - 2, self._max_cells - 2)
        minimum_goal_dist = 0.80 * self._max_cells #(at least that far)
        for e_x in range(1, self._max_cells, 2):
            for e_y in range(1, self._max_cells, 2):
                e_x = random.randint(0, (self._max_cells - 1) // 2 - 1) * 2 + 1
                e_y = random.randint(0, (self._max_cells - 1) // 2 - 1) * 2 + 1
                d_s_e = numpy.sqrt((e_x - s_x) ** 2 + (e_y - s_y) ** 2)
                if(d_s_e > minimum_goal_dist):
                    self._goal = (e_x, e_y)
                    break

        #Initialize the logics for prim based maze generation
        wall_dict = dict()
        path_dict = dict()
        rev_path_dict = dict()
        path_idx = 0
        for i in range(1, self._max_cells - 1):
            for j in range(1, self._max_cells - 1):
                if(self._cell_walls[i,j] > 0): # we will keep the axial point
                    wall_dict[i, j] = 0
                elif(self._cell_walls[i,j] == 0):
                    path_dict[i, j] = path_idx
                    rev_path_dict[path_idx] = [(i,j)]
                    path_idx += 1

        #Prim the wall until start and goal are connected
        #while path_dict[self._start] != path_dict[self._goal]:
        #Prim the wall until all points are connected
        while len(rev_path_dict) > 1:
            wall_list = list(wall_dict.keys())
            random.shuffle(wall_list)
            for i, j in wall_list:
                new_path_id = -1
                connected_path_id = dict()
                abandon_path_id = dict()
                max_duplicate = 1

                for d_i, d_j in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if(self.is_cell_valid(d_i, d_j) and self._cell_walls[d_i, d_j] < 1):
                        # calculate duplicate path id that might creat a loop
                        if path_dict[d_i, d_j] not in connected_path_id:
                            connected_path_id[path_dict[d_i, d_j]] = 1
                        else:
                            connected_path_id[path_dict[d_i, d_j]] += 1
                        if(connected_path_id[path_dict[d_i, d_j]] > max_duplicate):
                            max_duplicate = connected_path_id[path_dict[d_i, d_j]]

                        # decide the new path_id and find those to be deleted
                        if(path_dict[d_i, d_j] < new_path_id or new_path_id < 0):
                            if(new_path_id >= 0):
                                abandon_path_id[new_path_id] = (new_i, new_j)
                            new_path_id = path_dict[d_i, d_j]
                            new_i = d_i
                            new_j = d_j
                        elif(path_dict[d_i, d_j] != new_path_id): # need to be abandoned
                            abandon_path_id[path_dict[d_i, d_j]] = (d_i, d_j)
                if(len(abandon_path_id) >= 1 and max_duplicate < 2):
                    break
                if(len(abandon_path_id) >= 1 and max_duplicate > 1 and allow_loops):
                    break
                if(allow_loops and random.random() < 0.1):
                    break

            if(new_path_id < 0):
                continue
                        
            # add the released wall
            rev_path_dict[new_path_id].append((i,j))
            path_dict[i,j] = new_path_id
            self._cell_walls[i,j] = 0
            del wall_dict[i,j]

            # merge the path
            for path_id in abandon_path_id:
                rev_path_dict[new_path_id].extend(rev_path_dict[path_id])
                for t_i_o, t_j_o in rev_path_dict[path_id]:
                    path_dict[t_i_o,t_j_o] = new_path_id
                del rev_path_dict[path_id]

        #Paint the texture of passways to ground textures 
        for i in range(1, self._max_cells - 1):
            for j in range(1, self._max_cells - 1):
                if(self._cell_walls[i,j] < 1):
                    self._cell_texts[i,j] = 0

    def load_maze(self, cell_walls, cell_texts, start, goal):
        assert len(numpy.shape(cell_walls))==2 and numpy.shape(cell_walls)[0]==numpy.shape(cell_walls)[1] and cell_walls.shape==cell_texture.shape
        self._max_cells = numpy.shape(cell_walls)[0]
        self._start = start
        self._goal = goal
        self._cell_walls = cell_walls
        self._cell_texts = cell_texts

    def dump_maze(self):
        return self._cell_walls, self._cell_texts, self._start, self._goal

    def reset(self):
        self._goal_position = self.get_cell_center(self._goal)
        self._agent_pos = self.get_cell_center(self._start)
        self._agent_ori = PI_4
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
        if(goal_dist < 0.50): #reaching the goal
            done = True
        return done

    def render_init(self, view_size=480, god_view=True):
        print("initializing render...")
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._max_cells
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._god_view = god_view
        self._ori_size = 0.60 * self._pos_conversion
        self._view_size = view_size

        if(self._god_view):
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
            pygame.display.set_caption("RandomMazeRender")

    def render_update(self):
        #Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))
        #Paint God View
        if(self._god_view):
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
        return keys, done

    def movement_control(self, keys):
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
                self._cell_walls, self._cell_transparents, self._cell_texts, self._cell_size, self._texture_list,
                self._texture_ceil, self._wall_height, 1.0, self._max_vision_range, 0.20, 
                self._fol_angle, self._resolution_horizon, self._resolution_vertical)
        self._obs_surf = pygame.surfarray.make_surface(self.get_observation())
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
        return self._observation

if __name__=='__main__':
    env = MazeEnv(with_guidepost=True)
    env.gen_random_maze(max_cells=21, allow_loops=True)
    env.reset()
    done = False
    env.render_init(view_size=640, god_view=False)
    while not done:
        keys, key_done = env.render_update()
        tr, ws = env.movement_control(keys)
        done = env.do_action(tr, ws)
        pygame.time.delay(20)
        done = done or key_done
