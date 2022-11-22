# Introduction

Meta language model generates sequences of repeating random integers with noises, aims at facilitating researches in Lifelong In-Context Learning.
$MetaLM(V, l, e, L)$ data generator generates the sequence by the following steps:

- 1.Generating $l_r \sim Poisson(l)$ 
- 2.Randomly sampling a sequence $s$ of length $l_r$ composed of numbers between 1 and V
- 3.Generating $S$ by repeating $s$ until reaching the length of L
- 4.Disturbing the seuqeunce $S$ by randomly replacing the number in $S$ with wrong numbers or specific number 0

A meta language model should predict the undisturbed next integer from the disturbed sequence (by observing only the integers before the current step). The model ought to predict better and better at the rear part of the sequence.

#### Examples
A example of $MetaLM(V=50, l=4, e=0.15, L=25)$

The observed disturbed sequence:

44	39	44	27	44	39	48	27	44	39	48	27	44	39	17	27	44	39	48	27	44	39	48	0	44

The original undisturbed sequence to be predicted:

39	48	27	44	39	48	27	44	39	48	27	44	39	48	27	44	39	48	27	44	39	48	27	44	39

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
