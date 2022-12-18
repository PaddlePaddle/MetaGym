# Introduction

MetaMaze is a powerful and efficient simulator for navigation in a randomly generated maze. You may use MetaMaze to generate nearly unlimited configuration of mazes, and nearly unlimited different tasks. We use MetaMaze to facilitate researches in Meta-Reinforcement-Learning.

There are 3 types of mazes:

- **meta-maze-2D-v0** <br>
--- Observation space: its surrounding $(2n+1)\times(2n+1)$ (n specified by view_grid parameter) grids<br>
--- Action space: 4-D discrete N/S/W/E <br><br>
- **meta-maze-discrete-3D-v0** <br>
--- Observation space: RGB image of 3D first-person view. <br>
--- Action space: 4-D discrete TurnLeft/TurnRight/GoForward/GoBackward. <br><br>
- **meta-maze-continuous-3D-v0** <br>
--- Observation space: RGB image of 3D first-person view.<br>
--- Action space: 2-D continuous [Turn, Forward/Backward]<br><br>

Each type of mazes support 2 modes:

- **ESCAPE** mode <br>
--- Reach an unknown goal as soon as possible <br>
--- The goal is specified by task configuration <br>
--- Each step the agent receives reward of step_reward <br>
--- Acquire goal_reward when reaching the goal <br>
--- Episode terminates when reaching the goal <br><br>
- **SURVIVAL** mode <br>
--- The agent begins with initial_life specified by the task <br>
--- Food is generated at fixed grids specified by the task <br>
--- When agent reaches the food spot, its life is extended depending on the food <br>
--- When food is cosumed, it will be refreshed following a fixed periodicity <br>
--- The life slowly decreases with time, depeding on step_reward <br>
--- Episode terminates when life goes below 0 <br>
--- The total reward is the food being consumed <br>
--- The agent's current life is shown by a red bar at the top of its view in 3D mazes <br>
--- The agent's current life is shown in the center of the $(2n+1)\times(2n+1)$ in 2D mazes <br><br>

Demonstrations of 2D maze

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_2d.gif" width="600"/>

Demonstrations of 3D mazes

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_small.gif" width="600"/>

<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_huge.gif" width="600"/>

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

## Writing your own policy

Specifying action with your own (Meta RL) policy without relying on keyboards and rendering, check
```bash
python metagym/metamaze/test.py
```
