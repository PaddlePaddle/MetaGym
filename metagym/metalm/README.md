# Introduction

Meta language model generates sequences of repeating random integers with noises, aims at facilitating researches in Lifelong In-Context Learning.
$MetaLM(V, n, l, e, L)$ data generator generates the sequence by the following steps:

- 1.Generating elements $s_i \in S$ of length $l_i~Poisson(l), i\in[1,n]$ by random pick integers from $[1, V]$.
- 2.At iteration t, randomly sampling $s_t \in S$, disturb the seuqence $s_t$ acquring $\hat{s}_t$ by randomly replacing the number in $s$ with the other numbers or specific number 0. 
- 3.Contatenating $\hat{s}_t$ to $x$, iterate step 2 until $x$ reaching length of L, concatenating $s_t$ to $y$

A meta langauge model is required to predict $p(y_{l+1}|\hat_{s}_{t}, \hat_{s}_{t-1}, ..., \hat_{s}_{1}$

### Motivation

Each $x$ can be regarded as an unknown language that has $V$ embedding space, 

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

Import and create the meta language generator
```python
import gym
import metagym.metalm

env = gym.make("meta-lm-v0", V=50, l=50, e=0.15, L=4096)
```

## Generating unlimited data

```python
obs, demo_label = env.data_generator() # generate observation & label for one sample
batch_obs, batch_label = env.batch_generator(batch_size) # generate observations & labels for batch of sample (shape of [batch_size, L])
```
