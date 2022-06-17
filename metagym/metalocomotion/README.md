# Introduction

MetaLocomotion implement canonical gym bullet locomotion environments, but with diversed geometries. The repo is inspired by [PyBulletGym](https://github.com/benelot/pybullet-gym)
We provide a total of 368 ants and 368 humanoids, which have diverse limb lengths, sampled by multiplying the original limb length by random numbers.
Among those configurations, 256 of them are training set, 64 are testing set. The left 64 are extreme examples which we classified to out-of-distribution testing set.

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

loco_env = gym.make("meta-humanoids-v0") # Running meta humanoids
#loco_env = gym.make("meta-ants-v0") # Running meta ants
```

## Sampling Geometries

Use the following code to sample an unique geometry
```python
#Sample a task by specifying the configurations
task = loco_env.sample_task(
        task_type = "TRAIN" # we provide 256 geometries for training
        #task_type = "TEST" # we provide 64 geometries for testing
        #task_type = "OOD" # we provide 64 extreme testing cases
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

<img src="envs/assets/Ants_demo.png" width="2400"/>

<img src="envs/assets/Humanoids_demo.png" width="1200"/>
