# MetaMaze

MetaMaze is a powerful and efficient simulator for 3D navigation in a randomly generated maze, benchmarking meta learning algorithms. In MetaMaze you can specify different

#### Maze Architectures
#### Maze Scales
#### Cell Sizes
#### Wall Heights
#### Wall Textures
#### Agent Heights

to test the capability of your learning algorithm adapting to different challenging tasks.

# Demonstrations

A demonstration of maze cell_scale=15, cell_size=2, wall_height=3.2
<img src="envs/img/demo_maze_small.gif" width="600"/>

A demonstration of maze cell_scale=9, cell_size=5, wall_height=6.4
<img src="envs/img/demo_maze_huge.gif" width="600"/>

Run the following command to get a quick view and play of the game
```bash
python keyboard_play_demo.py
```

## Install

```bash
pip install rlschool
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/PaddlePaddle/RLSchool
cd RLSchool
pip install .
```

## Quick Start

A Quick Demonstration
```python
from rlschool import make_env
#Start a new meta environment with
maze_env = make_env("MetaMaze",
    enable_render=True # False if you do not need a render
    )

#Sample a task by specifying the configurations
task = maze_env.sample_task(
    cell_scale=15,  # Number of cells = cell_scale * cell_scale
    allow_loops=True,  # Whether loops are allowed
    cell_size=2.0, # specifying the size of each cell
    wall_height=3.2, # specifying the height of the wall
    agent_height=1.6 # specifying the height of the agent
    )
    
#Set the task configuration to the meta environment
maze_env.set_task(task)

#Start the task
done = False
while not done:
    action = # Specify your policy, deciding the turn(LEFT/RIGHT) and the walk speed (FORWARD/BACKWARD)
    observation, reward, done, info = maze_env.do_action(action)
    maze_env.render()
```

## Citation

```txt
@misc{MetaMaze,
    author = {Fan Wang},
    title = {{MetaMaze: Efficient 3D Navigation Simulator Benchmarking Meta-learning}},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/metamaze}},
}
```
