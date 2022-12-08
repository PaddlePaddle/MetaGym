# Introduction

This Environment can be used to generate unlimited different MAB tasks.<br>
The arbitrary multi-armed bandits (MAB) tasks can be solved using Meta-Reinforcement-Learning, see:

```
Mishra, Nikhil, et al. "A Simple Neural Attentive Meta-Learner." International Conference on Learning Representations. 2018.
```

# Install

```bash
pip install metagym
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/PaddlePaddle/MetaGym
cd MetaGym
pip install .
```

# Quick Start

## Import

Import and create the bandits environment with 
```python
import gym
import metagym.bandits

env = gym.make("bandits-v0", arms=10, max_steps=4096)
```

## Sampling an arbitrary Bandits task
```python
task = env.sample_task(
        distribution_settings="Classical",  # allowing Classical / Uniform / Gaussian
        mean=0.50, # mean expected gain of all the arms
        dev=0.05 # variance of expected gain of the arms
        )
env.set_task(task)
env.reset()
```

## Running a demo of thompson sampling with
```script
python demo_thompson_sampling.py
```
