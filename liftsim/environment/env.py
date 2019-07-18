# wrapper part modified from
# https://github.com/openai/gym/blob/master/gym/core.py

from environment.mansion.person_generators.generator_proxy import set_seed
from environment.mansion.person_generators.generator_proxy import PersonGenerator
from environment.mansion.mansion_config import MansionConfig
from environment.mansion.utils import ElevatorState, MansionState
from environment.mansion.mansion_manager import MansionManager
from environment.wrapper_utils import obs_dim, act_dim, mansion_state_preprocessing
from environment.wrapper_utils import action_idx_to_action, action_to_action_idx

NoDisplay = False
try:
    from environment.animation.rendering import Render
except Exception as e:
    NoDisplay = True

import argparse
import configparser
import random
import sys


class environmentEnv():
    '''
    environmentation Environment
    '''

    def __init__(self, config_file='./config.ini'):
        file_name = config_file

        config = configparser.ConfigParser()
        config.read(file_name)

        time_step = float(config['Configuration']['RunningTimeStep'])
        assert time_step <= 1, 'RunningTimeStep in config.ini must be less than 1 in order to ensure accuracy'

        # Readin different person generators
        gtype = config['PersonGenerator']['PersonGeneratorType']
        person_generator = PersonGenerator(gtype)
        person_generator.configure(config['PersonGenerator'])

        self._config = MansionConfig(
            dt=time_step,
            number_of_floors=int(config['MansionInfo']['NumberOfFloors']),
            floor_height=float(config['MansionInfo']['FloorHeight'])
        )

        if('LogLevel' in config['Configuration']):
            assert config['Configuration']['LogLevel'] in ['Debug', 'Notice', 'Warning'],\
                        'LogLevel must be one of [Debug, Notice, Warning]'
            self._config.set_logger_level(config['Configuration']['LogLevel'])
        if('Lognorm' in config['Configuration']):
            self._config.set_std_logfile(config['Configuration']['Lognorm'])
        if('Logerr' in config['Configuration']):
            self._config.set_err_logfile(config['Configuration']['Logerr'])

        self._mansion = MansionManager(
            int(config['MansionInfo']['ElevatorNumber']),
            person_generator,
            self._config,
            config['MansionInfo']['Name']
        )

        self.mansion_attr = self._mansion.attribute
        self.elevator_num = self.mansion_attr.ElevatorNumber
        self.observation_space = obs_dim(self.mansion_attr)
        self.action_space = act_dim(self.mansion_attr)

        self.viewer = None

    def seed(self, seed=None):
        set_seed(seed)

    def step(self, action):
        time_consume, energy_consume, given_up_persons = self._mansion.run_mansion(action)
        reward = - (time_consume + 0.01 * energy_consume +
                    1000 * given_up_persons) * 1.0e-5
        info = {'time_consume':time_consume, 'energy_consume':energy_consume, 'given_up_persons': given_up_persons}
        return (self._mansion.state, reward, False, info)

    def reset(self):
        self._mansion.reset_env()
        return self._mansion.state

    def render(self):
        if self.viewer is None:
            if NoDisplay:
                raise Exception('[Error] Cannot connect to display screen. \
                    \n\rYou are running the render() functoin on a manchine that does not have a display screen')
            self.viewer = Render(self._mansion)
        self.viewer.view()

    def close(self):
        pass

    @property
    def attribute(self):
        return self._mansion.attribute

    @property
    def state(self):
        return self._mansion.state

    @property
    def statistics(self):
        return self._mansion.get_statistics()

    @property
    def log_debug(self):
        return self._config.log_notice

    @property
    def log_notice(self):
        return self._config.log_notice

    @property
    def log_warning(self):
        return self._config.log_warning

    @property
    def log_fatal(self):
        return self._config.log_fatal


class Wrapper(environmentEnv):
    def __init__(self, env):
        self.env = env
        self._mansion = env._mansion
        self.elevator_num = env.elevator_num
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.viewer = env.viewer

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    
class RewardWrapper(Wrapper):
    pass

class ActionWrapper(Wrapper):
    def step(self, action):
        return self.env.step([action_idx_to_action(a, self.action_space) for a in action])

class ObservationWrapper(Wrapper):
    def reset(self):
        self.env.reset()
        return mansion_state_preprocessing(self._mansion.state)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return (mansion_state_preprocessing(observation), reward, done, info)

    @property
    def state(self):
        return mansion_state_preprocessing(self._mansion.state)
