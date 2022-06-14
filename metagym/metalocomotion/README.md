# Introduction

MetaLocomotion implement canonical gym bullet locomotion environments, but with diversed geometries. For instance, in meta ants, different tasks differs in the ant's thigh and thin lengths in its four legs.

Currently we support
#### Meta Ants
#### Meta Humanoids

# Install

```bash
pip install metagym[metalocomotion]
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/PaddlePaddle/MetaGym
cd MetaGym
pip install .[metalocomotion]
```

# Quick Start

## Import

Import and create the meta maze environment with 
```python
import gym
import metagym.metalocomotion

loco_env = gym.make("meta-humanoids-v0", enable_render=True) # Running meta humanoids
#loco_env = gym.make("meta-ants-v0", enable_render=True) # Running meta ants
```

## Sampling Geometries

Use the following code to sample an unique geometry
```python
#Sample a task by specifying the configurations
task = loco_env.sample_task(
        task_type = "TRAIN" # we provide 256 geometries for training
        #task_type = "TEST" # we provide 64 geometries for testing
        #task_type = "OOD" # we provide 64 Out-of-distribution geometries, which is exceptionally harder
        )
```

## Running Locomotion
```python
#Set the task configuration to the meta environment
loco_env.set_task(task)
loco_env.reset()

#Start the task
done = False
while not done:
    action = loco_env.action_space.sample() 
    observation, reward, done, info = loco_env.step(action)
    loco_env.render()
```

# Examples of Different Geometries
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/ant_exp.png" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/ants_1.gif" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/ants_2.gif" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/humanoid_exp.png" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/humanoids_1.gif" width="600"/>
<img src="https://github.com/benchmarking-rl/PARL-experiments/blob/master/MetaGym/humanoids_2.gif" width="600"/>
