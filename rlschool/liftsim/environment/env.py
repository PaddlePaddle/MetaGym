#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import set_seed
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.mansion_manager import MansionManager
from rlschool.liftsim.environment.mansion.utils import ElevatorAction


NoDisplay = False
try:
    from rlschool.liftsim.environment.animation.rendering import Render
except Exception as e:
    NoDisplay = True

import argparse
import configparser
import random
import sys
import os


class LiftSim():
    """
    environmentation Environment
    """
    def __init__(self, config_file=os.path.join(os.path.dirname(__file__)+'/../config.ini'), **kwargs):
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

        self.viewer = None

    def seed(self, seed=None):
        set_seed(seed)

    def step(self, action):
        assert type(action) is list, "Type of action should be list"
        assert len(action) == self._mansion.attribute.ElevatorNumber*2, \
            "Action is supposed to be a list with length ElevatorNumber * 2"
        action_tuple = []
        for i in range(self._mansion.attribute.ElevatorNumber):
            action_tuple.append(ElevatorAction(action[i*2], action[i*2+1]))
        time_consume, energy_consume, given_up_persons = self._mansion.run_mansion(action_tuple)
        reward = - (time_consume + 5e-4 * energy_consume +
                    300 * given_up_persons) * 1.0e-4
        info = {'time_consume':time_consume, 'energy_consume':energy_consume, 'given_up_persons': given_up_persons}
        return (self._mansion.state, reward, False, info)

    def reset(self):
        self._mansion.reset_env()
        return self._mansion.state

    def render(self):
        if self.viewer is None:
            if NoDisplay:
                raise Exception('[Error] Cannot connect to display screen. \
                    \n\rYou are running the render() function on a manchine that does not have a display screen')
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
