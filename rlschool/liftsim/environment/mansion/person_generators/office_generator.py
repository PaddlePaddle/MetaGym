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
# Generating Persons for typical office buildings:
# Very high pedestrian flow during the morning and the evening
# Average amount of pedestrian flow in other cases

import sys
import random
from rlschool.liftsim.environment.mansion.utils import EPSILON
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.person_generators.person_generator import PersonGeneratorBase


class OfficePersonGenerator(PersonGeneratorBase):
    """
    Basic Generator Class
    Generates Random Person from Random Floor, going to random other floor
    Uniform distribution in floor number, target floor number etc
    """

    def configure(self, configuration):
        self._particle_number = int(configuration['ParticleNumber'])
        self._particle_interval = float(configuration['GenerationInterval'])
        self._random_interval = list(map(lambda x: 3600.0 * self._particle_number / max(
            float(x), EPSILON), self._config['RandomFreqPattern'].split(',')))
        self._upstairs_interval = list(map(lambda x: 3600.0 * self._particle_number / max(
            float(x), EPSILON), self._config['UpstairsFreqPattern'].split(',')))
        self._downstairs_interval = list(map(lambda x: 3600.0 * self._particle_number / max(
            float(x), EPSILON), self._config['DownstairsFreqPattern'].split(',')))
        assert len(self._random_interval) == 24
        assert len(self._upstairs_interval) == 24
        assert len(self._downstairs_interval) == 24
        self._cur_id = 0

    def _weight_generator(self):
        return random.normalvariate(50, 10)

    def generate_person(self):
        """
        Generate Random Persons from Poisson Distribution
        Args:
          None
        Returns:
          List of Random Persons
        """
        ret_persons = []
        time_interval = self._config.raw_time - self._last_generate_time
        cur_hour = self._config.world_time
        for i in range(self._particle_number):
            if(random.random() < time_interval / self._random_interval[cur_hour]):
                random_source_floor = random.randint(1, self._floor_number)
                random_target_floor = random.randint(1, self._floor_number)
                while random_source_floor == random_target_floor:
                    random_source_floor = random.randint(1, self._floor_number)
                    random_target_floor = random.randint(1, self._floor_number)
                random_weight = self._weight_generator()

                ret_persons.append(
                    PersonType(
                        self._cur_id,
                        random_weight,
                        random_source_floor,
                        random_target_floor,
                        self._config._current_time))
                self._cur_id += 1

            if(random.random() < time_interval / self._upstairs_interval[cur_hour]):
                source_floor = 1
                target_floor = random.randint(2, self._floor_number)
                random_weight = self._weight_generator()

                ret_persons.append(
                    PersonType(
                        self._cur_id,
                        random_weight,
                        source_floor,
                        target_floor,
                        self._config._current_time))
                self._cur_id += 1

            if(random.random() < time_interval / self._downstairs_interval[cur_hour]):
                source_floor = random.randint(2, self._floor_number)
                target_floor = 1
                random_weight = self._weight_generator()

                ret_persons.append(
                    PersonType(
                        self._cur_id,
                        random_weight,
                        source_floor,
                        target_floor,
                        self._config._current_time))
                self._cur_id += 1

        #float_time = formulated_time.Hour + float(formulated_time.Min)/60.0 + float(formulated_time.Sec) / 3600.0

        self._last_generate_time = self._config.raw_time
        return ret_persons
