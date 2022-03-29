# Introduction

MetaMaze is a powerful and efficient simulator for navigation in a randomly generated maze. We support 2D-Navigation and 3D-Navigation. You may use mazes of different architectures and textures as tasks for benchmarking Meta-Learning algorithms.

# Install

```bash
pip install metagym[metamaze]
```

#### For local installation, execute following commands:

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

maze_env = gym.make("meta-maze-3D-v0", enable_render=True) # Running a 3D Maze
#maze_env = gym.make("meta-maze-2D-v0", enable_render=True) # Running a 2D Maze
```

## Maze Generation

Use the following code to generate a random maze
```python
#Sample a task by specifying the configurations
task = maze_env.sample_task(
    cell_scale=15,  # Number of cells = cell_scale * cell_scale
    allow_loops=False,  # Whether loops are allowed
    crowd_ratio=0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
    cell_size=2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height=3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height=1.6, # specifying the height of the agent, only valid for 3D mazes
    view_grids=1 # specifiying the observation region for the agent, only valid for 2D mazes
    )
```

## Running 2D Mazes
```python
#Set the task configuration to the meta environment
maze_env.set_task(task)
maze_env.reset()

#Start the task
done = False
while not done:
    #  The action space is discrete actions specifying UP/DOWN/LEFT/RIGHT
    action = maze_env.action_space.sample() 
    #  The observation being 3 * 3 numpy array, the observation of its current neighbours
    #  Reward is set to be 20 when arriving at the goal, -0.1 for each step taken
    #  Done = True when reaching the goal or maximum steps (200 as default)
    observation, reward, done, info = maze_env.step(action)
    maze_env.render()
```

## Running 3D Mazes
```python
#Set the task configuration to the meta environment
maze_env.set_task(task)
maze_env.reset()

#Start the task
done = False
while not done:
    #  The action space is continuous Boxes in (-1, 1) deciding the turning (LEFT/RIGHT) and the walking speed (FORWARD/BACKWARD)
    action = maze_env.action_space.sample() 
    #  The observation being RGB picture of W * H * 3
    #  Reward is set to be 200 when arriving at the goal, -0.1 for each step taken
    #  Done = True when reaching the goal or maximum steps of 1000
    observation, reward, done, info = maze_env.do_action(action)
    maze_env.render()
```

# Keyboard Demonstrations

## 2D Mazes Demonstration

For a demonstration of keyboard controlled 2D mazes, run
```bash
python metagym/metamaze/keyboard_play_demo_2d.py
```
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_2d.gif" width="600"/>

## 3D Mazes Demonstration

For a demonstration of keyboard controlled 3D mazes, run
```bash
python metagym/metamaze/keyboard_play_demo_3d.py
```
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_small.gif" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/demo_maze_huge.gif" width="600"/>

## Writing your own policy

Specifying action with your own (RL) policy without relying on keyboards and rendering, check
```bash
python metagym/metamaze/test.py
```

## Citation

```txt
@misc{MetaMaze,
    author = {Fan Wang},
    title = {{MetaMaze: Efficient 3D Navigation Simulator Benchmarking Meta-learning}},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/PaddlePaddle/MetaGym/tree/master/metagym/metamaze}},
}
```
