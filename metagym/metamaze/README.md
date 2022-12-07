# Introduction

MetaMaze is a powerful and efficient simulator for navigation in a randomly generated maze. We support 2D-Navigation and 3D-Navigation. You may use mazes of different architectures and textures as tasks for benchmarking Meta-Learning algorithms.

There are three types of mazes:

- meta-maze-2D-v0 : 2D mazes, the observation space is its surrounding $(2n+1)\times(2n+1)$ grids, The action space is discrete N/S/W/E
- meta-maze-discrete-3D-v0 : mazes with 3D view. Wall is marked with different textures. The action space is discrete TurnLeft/TurnRight/GoForward/GoBackward. The observation space is RGB image.
- meta-maze-continuous-3D-v0 : mazes with 3D view and continuous control on its orientation and moving speed. The action is continuous (turn, forward_speed). The observation space is RGB image.

Each of the mazes are supported with 2 type of tasks

- ESCAPE mode: the agent tries to reach an unknown goal as soon as possible, an episode is over when reach the goal.
- SURVIVAL mode: the agent consumes its life at each times step, it must find food spot in the maze to supply its life. The food spot refreshes periodically. An episode is over when life goes down to zero. In SURVIVAL mode, the current life of the agent is shown by the red bar in the top of RGB image in 3D mazes, and denoted by the value in the center of $(2n+1)\times(2n+1)$ observation in 2D mazes.

# Install

```bash
pip install metagym[metamaze]
```

#### For local installation, execute the following commands:

```bash
git clone https://github.com/PaddlePaddle/MetaGym
cd MetaGym
pip install .[metamaze]
```

# Quick Start

## Import

Import and create the meta maze environment with 
```python
import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler

maze_env_3D_Cont = gym.make("meta-maze-continuous-3D-v0", enable_render=True, task_type="SURVIVAL") # Running a continuous 3D Maze with SURVIVAL task
maze_env_3D_Disc = gym.make("meta-maze-discrete-3D-v0", enable_render=True, task_type="ESCAPE") # Running a discrete 3D Maze with ESCAPE task
maze_env_2D = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE") # Running a 2D Maze with ESCAPE task
```

## Maze Generation

Use the following code to generate a random maze
```python
#Sample a task by specifying the configurations
task = MazeTaskSampler(
    n            = 15,  # Number of cells = n*n
    allow_loops  = False,  # Whether loops are allowed
    crowd_ratio  = 0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
    cell_size    = 2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height  = 3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height = 1.6, # specifying the height of the agent, only valid for 3D mazes
    view_grid    = 1, # specifiying the observation region for the agent, only valid for 2D mazes
    step_reward  = -0.01, # specifying punishment in each step in ESCAPE mode, also the reduction of life in each step in SURVIVAL mode
    goal_reward  = 1.0, # specifying reward of reaching the goal, only valid in ESCAPE mode
    initial_life = 1.0, # specifying the initial life of the agent, only valid in SURVIVAL mode
    max_life     = 2.0, # specifying the maximum life of the agent, acquiring food beyond max_life will not lead to growth in life. Only valid in SURVIVAL mode
    food_density = 0.01,# specifying the density of food spot in the maze, only valid in SURVIVAL mode
    food_interval= 100, # specifying the food refreshing periodicity, only valid in SURVIVAL mode
    )
```

## Running Mazes
```python
#Set the task configuration to the meta environment
maze_env.set_task(task)
maze_env.reset()

#Start the task
done = False
while not done:
    action = maze_env.action_space.sample() 
    observation, reward, done, info = maze_env.step(action)
    maze_env.render()
```

# Keyboard Demonstrations

## 2D Mazes Demonstration

For a demonstration of keyboard controlled 2D mazes, run
```bash
python metagym/metamaze/keyboard_play_demo_2d.py
```
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_2d.gif" width="600"/>

## 3D Discrete Mazes Demonstration

For a demonstration of keyboard controlled discrete 3D mazes, run
```bash
python metagym/metamaze/keyboard_play_demo_discrete_3d.py
```

## 3D Continuous Mazes Demonstration

For a demonstration of keyboard controlled 3D mazes, run
```bash
python metagym/metamaze/keyboard_play_demo_continuous_3d.py
```
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_small.gif" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_huge.gif" width="600"/>

## Writing your own policy

Specifying action with your own (Meta RL) policy without relying on keyboards and rendering, check
```bash
python metagym/metamaze/test.py
```
