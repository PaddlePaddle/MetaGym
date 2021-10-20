# Introduction

Navigator2D is a simple benchmarking environment for meta reinforcement learning. The tasks are to navigate a two-wheeled robot to randomly generated goals in 2-D space. We assume that there is a signal transmitter on the goal and a receiver on the robot. The robot observes the signal intensity decided by A<sub>t</sub>=A<sub>0</sub> - k * log(d<sub>t</sub>/d<sub>0</sub>) + &epsilon; (Friis et al., 1946, A note on a simple transmission formula), where d<sub>t</sub> is the current distance between the robot and the goal, &epsilon; is the white noise in the observation. Each task has random goal and transmission patterns dependent on A<sub>0</sub>, k. The action is the rotation speed of its two wheels that controls the orientation and velocity of the robot. The observation is one dimensional (A<sub>t</sub>). The reward at each step is r<sub>t</sub>=- k * d<sub>t</sub>, (k=0.1), an episode terminates when the robot approaches the goal or maximum steps are reached. 

# Install

```bash
pip install rlschool[navigator2d]
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/PaddlePaddle/RLSchool
cd RLSchool
pip install .[navigator2d]
```

# Quick Start

## Import

Import and create the meta maze environment with 
```python
import gym
import rlschool.navigator2d

env = gym.make("navigator-wr-2D-v0", enable_render=True)
```

## Task Generation

Use the following code to sample a random task. A random task specifies a random goal, and a signal transmission pattern

```python
#Sample a task by specifying the configurations
task = env.sample_task()
```

## Demonstration
```python
env.set_task(env.sample_task())
env.reset()
done = False
while not done:
    # action is continuous 2 dimensional
    action = env.action_space.sample()
    # obs is one dimensional
    obs, r, done, info = env.step(action)
```

# Keyboard Demonstrations

For a demonstration of keyboard controlled 2D mazes, run
```bash
python rlschool/navigator2d/keyboard_control_demo.py
```
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/RLSchool/demo_navigator_2d.gif" width="600"/>
