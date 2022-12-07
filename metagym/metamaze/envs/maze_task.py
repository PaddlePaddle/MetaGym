"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from collections import namedtuple
from numpy import random as npyrnd
from numpy.linalg import norm

class MazeTaskManager(object):

    # Configurations that decides a specific task
    TaskConfig = namedtuple("TaskConfig", ["start", "goal", "cell_walls", "cell_texts", 
        "cell_size", "wall_height", "agent_height", "initial_life", "max_life",
        "step_reward", "goal_reward", "food_rewards", "food_interval"])

    def __init__(self, texture_dir, verbose=False):
        pathes = os.path.split(os.path.abspath(__file__))
        texture_dir = os.sep.join([pathes[0], texture_dir])
        texture_files = os.listdir(texture_dir)
        texture_files.sort()
        grounds = [None]
        for file_name in texture_files:
            if(file_name.find("wall") >= 0):
                grounds.append(pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name]))))
            if(file_name.find("ground") >= 0):
                grounds[0] = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("ceil") >= 0):
                self.ceil = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("arrow") >= 0):
                self.arrow = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
        self.grounds = numpy.asarray(grounds, dtype="float32")
        self.verbose = verbose

    @property
    def n_texts(self):
        return self.grounds.shape[0]

    def sample_task(self,
            n=15, 
            allow_loops=True, 
            cell_size=2.0, 
            wall_height=3.2, 
            agent_height=1.6,
            step_reward=-0.01,
            goal_reward=None,
            food_reward=0.50,
            initial_life=1.0,
            max_life=2.0,
            food_density=0.010,
            food_interval=100,
            crowd_ratio=0.0):
        # Initialize the maze ...
        assert n > 6, "Minimum required cells are 7"
        assert n % 2 != 0, "Cell Numbers can only be odd"
        if(self.verbose):
            print("Generating an random maze of size %dx%d, with allow loops=%s, crowd ratio=%f"%(n, n, allow_loops, crowd_ratio))
        cell_walls = numpy.ones(shape=(n, n), dtype="int32")
        cell_texts = numpy.random.randint(1, self.n_texts, size=(n, n))

        # Dig the initial positions
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                cell_walls[i,j] = 0
        #Randomize a start, around the bottom-left
        s_x = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
        s_y = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
        start = (s_x, s_y)

        #Randomize a goal (not quite efficiently)
        goal = (n - 2, n - 2)
        minimum_goal_dist = 0.45 * n #(at least that far)
        for e_x in range(1, n, 2):
            for e_y in range(1, n, 2):
                e_x = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
                e_y = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
                d_s_e = numpy.sqrt((e_x - s_x) ** 2 + (e_y - s_y) ** 2)
                if(d_s_e > minimum_goal_dist):
                    goal = (e_x, e_y)
                    break

        #Initialize the logics for prim based maze generation
        wall_dict = dict()
        path_dict = dict()
        rev_path_dict = dict()
        path_idx = 0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i,j] > 0): # we will keep the axial point
                    wall_dict[i, j] = 0
                elif(cell_walls[i,j] == 0):
                    path_dict[i, j] = path_idx
                    rev_path_dict[path_idx] = [(i,j)]
                    path_idx += 1

        #Prim the wall until start and goal are connected
        #while path_dict[start] != path_dict[goal]:
        #Prim the wall until all points are connected
        max_cell_walls = numpy.product(cell_walls[1:-1, 1:-1].shape)
        while len(rev_path_dict) > 1 or (allow_loops and numpy.sum(cell_walls[1:-1, 1:-1]) > max_cell_walls * crowd_ratio):
            wall_list = list(wall_dict.keys())
            random.shuffle(wall_list)
            for i, j in wall_list:
                new_path_id = -1
                connected_path_id = dict()
                abandon_path_id = dict()
                max_duplicate = 1

                for d_i, d_j in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if((d_i > 0  and d_i < n and d_j > 0 and d_j < n)
                            and cell_walls[d_i, d_j] < 1):
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
                if(allow_loops and len(rev_path_dict) < 2 and random.random() < 0.2):
                    break

            if(new_path_id < 0):
                continue
                        
            # add the released wall
            rev_path_dict[new_path_id].append((i,j))
            path_dict[i,j] = new_path_id
            cell_walls[i,j] = 0
            del wall_dict[i,j]

            # merge the path
            for path_id in abandon_path_id:
                rev_path_dict[new_path_id].extend(rev_path_dict[path_id])
                for t_i_o, t_j_o in rev_path_dict[path_id]:
                    path_dict[t_i_o,t_j_o] = new_path_id
                del rev_path_dict[path_id]

        #Paint the texture of passways to ground textures 
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i,j] < 1):
                    cell_texts[i,j] = 0

        #Calculate goal reward, default is - n sqrt(n) * step_reward
        assert step_reward < 0, "step_reward must be < 0"
        if(goal_reward is None):
            def_goal_reward = - numpy.sqrt(n) * n * step_reward
        else:
            def_goal_reward = goal_reward
        assert def_goal_reward > 0, "goal reward must be > 0"

        # Generate food points
        food_rewards = numpy.clip(numpy.random.rand(n, n) * food_reward, 0.10, food_reward)
        food_rewards *= 1.0 - cell_walls
        exp_food = (n - 1) * (n - 1) * food_density
        while(numpy.sum(food_rewards) > exp_food): 
            food_rewards *= (numpy.random.rand(n, n) < 0.90).astype("float32")
        food_interval = food_interval * (food_rewards > 1.0e-3).astype("int32")

        return MazeTaskManager.TaskConfig(
                start=start,
                goal=goal,
                cell_walls=cell_walls,
                cell_texts=cell_texts,
                cell_size=cell_size,
                step_reward=step_reward,
                goal_reward=def_goal_reward,
                wall_height=wall_height,
                agent_height=agent_height,
                initial_life=initial_life,
                max_life=max_life,
                food_rewards=food_rewards,
                food_interval=food_interval
                )

MAZE_TASK_MANAGER=MazeTaskManager("img")
MazeTaskSampler = MAZE_TASK_MANAGER.sample_task
