English | [简体中文](./README_zh.md)

# LiftSim

LiftSim is a light-weight elevator simulator

Please consider to cite this environment if it can help your research.

```txt
@misc{LiftSim,
    author = {Fan Wang, Bo Zhou, Yunxiang Li, Kejiao Li},
    title = {{LiftSim: a configurable lightweight simulator of elevator systems}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/liftsim}},
}
```

<img src="demo_image.gif" width="400"/>


## Install

```python
pip install rlschool
```

#### For local installation, execute following commands:

```python
git clone https://github.com/PaddlePaddle/RLSchool
cd RLSchool
pip install .
```

## Quick Start

Liftsim environment follows the standard gym APIs to create, run, render and close an environment.

- reset(self): reset the environment to intial state, returns [observation](#Observation)
- step(self, action): takes action as input, returns the [observation](#Observation) of the next step, [reward](#Reward), done, info. Every step in Elevator simulation takes an actual time step of 0.5 seconds.
    - done: always False, i.e., won't stop until it is done.
    - info: a dictionary inlcuding the total wait time "time_consume" (float), consumption of energy "energy_consume" (float), number of customers who abandoned elevators because of waiting for too long "give_up_persons" (int). Refer to [Reward](#Reward) for details
- render(self): render one frame,

```python
# We show a simple example to start LiftSim here
from rlschool import make_env

env = make_env('LiftSim')
observation = env.reset()
action = [2, 0, 4, 0, 7, 0, 10, 0]
for i in range(100):
    env.render()    # use render to show animation
    next_obs, reward, done, info = env.step(action)
```

### Action

Action must be a list with length of 2\*n, which represent the command toward n different elevators (the default configuration has elevators n=4). Each command takes two elements
- The first element represent the DispatchTarget, which lies between 1 and MaxFloorNumber. Also DispatchTarget can either take a negative value -1, which represents that the previously specified DispatchTarget is not changed, or it can take the value of 0, which orders the elevator cart to stop ASAP.
- The second element represent the elevator moving direction after reaching the target. It may be -1 (Downwards), 0 (no direction), and 1 (upwards).

<img src="elevator_indicator.png" width="400"/>


## Observation

The observation includes the following structure

- MansionState：namedtuple, represents the overall states of all elevators

|Name                      |name                  |Description    |
|--------------------------|----------------------|---------------|
|ElevatorStates            |List of ElevatorState |A list of state of each elevator|
|RequiringUpwardFloors     |List of int           |Floors where there are customers requiring uplift|
|RequiringDownwardFloors   |List of int           |Floors where there are customers requiring downlift|


- ElevatorState：namedtuple, state of an elevator

| Name                    | Type    | Description                                |
| :----------------------:| :-----: | :----------------------------------------: |
| Floor                   | float   | Current floor where elevator locates         |
| MaximumFloor            | int     | The maximum floor of the current mansion     |
| Velocity                | float   | The current velocity of the elevator         |
| MaximumSpeed            | float   | The maximum speed allowed for the elevator cart  |
| Direction               | int     | The moving direction of the elevator cart, -1/0/1          |
| DoorState               | float   | The open ratio of the cart door       |
| CurrentDispatchTarget   | int     | The current specified dispatch target |
| DispatchTargetDirection | int     | The current specified dispatched direction  |
| LoadWeight              | float   | The current load of the elevator cart (kg)    |
| MaximumLoad             | float   | The maximum load of the elevator cart (kg)    |
| ReservedTargetFloors    | list    | The current required floors of the elevator cart |
| OverloadedAlarm         | float   | Whether the elevator is overloaded             |
| DoorIsOpening           | boolean | Whether the cart door is completely open       |
| DoorIsClosing           | Boolean | Whether the cart door is completely closed     |


## Examples

We hereby present a demonstration of dispatching an elevator system with Deep Q Network [Example][demo]

## Reward

The reward is calculated from three components:

- time_consume: the waiting time of all customers accumulated in a timestep, in seconds
- energy_consume: the energy comumed in a time step by the elevators, in J
- given_up_persons: number of people who give up waiting for the elevators because the waiting time is out of range.

We provide two different reward settings
### Economic Settings

```python
reward = - (time_consume + 0.01 * energy_consume + 100 * given_up_persons) * 1e-4
```

### Customer-Oriented Settings

```python
reward = - (time_consume + 5e-4 * energy_consume + 300 * given_up_persons) * 1e-4
```

## Configurations

LiftSim is a configurable elavator simulator
You can change the customer generation pattern by modifying [PersonGenerator] in [config.ini][config]
Also you may create your own unique pattern

### A uniform random customer generator

```ini
[PersonGenerator]
PersonGeneratorType = UNIFORM
ParticleNumber = 12
GenerationInterval = 150
```

### Curstom customer generator

```ini
[PersonGenerator]
PersonGeneratorType = CUSTOM
CustomDataFile = mansion_flow.npy
```

You may change mansion_flow.npy to your own customer flow data

## Related resources
[gym]: https://gym.openai.com/
[demo]: https://github.com/PaddlePaddle/PARL/tree/r1.3/examples/LiftSim_baseline/DQN
[submit]: https://aistudio.baidu.com/aistudio/competition/detail/11
[submit_folder]: https://github.com/Banmahhhh/RLSchool/blob/master/rlschool/liftsim/submit_folder.zip
[config]: https://github.com/PaddlePaddle/RLSchool/blob/master/rlschool/liftsim/config.ini
