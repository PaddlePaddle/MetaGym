# Introduction

Meta language model generates sequences of repeating random integers with noises, aims at facilitating researches in Lifelong In-Context Learning.
$MetaLM(V, n, l, e, L)$ data generator generates the sequence by the following steps:

- 1.Generating elements $s_i \in S$ of length $l_i \sim Poisson(l), i\in[1,n]$ by random pick integers from $[1, V]$.
- 2.At iteration t, randomly sampling $s_t \in S$, disturb the seuqence $s_t$ acquring $\bar{s}_t$ by randomly replacing the number in $s$ with the other numbers or specific number 0. 
- 3.Contatenating $\bar{s}_t$ to $x$, iterate step 2 until $x$ reaching length of L, concatenating $s_t$ to $y$

A meta langauge model:  $p(y_{l+1} \| x_{l}, x_{l-1}, ..., x_{1})$;
The meta language model should be doing better and better as the $l$ increases;

### Motivation

Each $x$ can be regarded as an unknown language composed of $V$ tokens. Its complexity is described by $n$ and $l$. Unlike the pre-trained language model that has effectively memorize the pattern in its parameters, in this dataset, as the elements of $x$ is randomly generated, the model can not possibly predict $y_{l+1}$ by using only the short term context, but has to depend on long-term repeating to correctly memorize the usage of the language. <br>

Although we refer to this model as meta language model, we understand it is a relatively simplified version of language, since a real language (e.g., natural langauge, programming language) can not be totally random. Still, this dataset can be used as valuable benchmarks for long term memory and lifelong In-Context learning. <br>

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

env = gym.make("meta-lm-v0", V=64, n=10, l=64, e=0.10, L=4096)
```

## Generating unlimited data

```python
obs, demo_label = env.data_generator() # generate observation & label for one sample
batch_obs, batch_label = env.batch_generator(batch_size) # generate observations & labels for batch of sample (shape of [batch_size, L])
```
