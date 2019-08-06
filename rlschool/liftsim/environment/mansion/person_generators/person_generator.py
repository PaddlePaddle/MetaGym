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

# Author: Fan Wang (wang.fan@baidu.com)
#
# Generators base class

import sys
import random
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig


class PersonGeneratorBase(object):
    """
    Basic Generator Class
    Generates Random Person from Random Floor, going to random other floor
    """

    def __init__(self):
        pass

    def configure(self, configuration):
        pass

    def link_mansion(self, mansion_config):
        self._config = mansion_config
        self._floor_number = self._config.number_of_floors
        self._last_generate_time = self._config.raw_time

    def generate_person(self, time):
        """
        Generate Random Persons from Poisson Distribution
        Args:
          None
        Returns:
          List of Random Persons
        """
        raise NotImplementedError()
