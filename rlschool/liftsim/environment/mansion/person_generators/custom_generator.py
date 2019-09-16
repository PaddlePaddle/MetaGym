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

import os
import sys
import random
import numpy as np
from rlschool.liftsim.environment.mansion.utils import EPSILON
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.person_generators.person_generator import PersonGeneratorBase


class CustomGenerator(PersonGeneratorBase):
    '''
    A customized generator by reading person flow data from data file.
    Customized Generator randomly generates human weights, but the source floor and target floor is generated
      according to the probability specified by the data file.
    The data file include statstics of pedestrian flow in each time interval
    Column 1: start of time interval
    Column 2: number of pedestrians flow in
    Column 3: number of pedestrians flow out
    '''

    def configure(self, configuration):
        self._data_file = os.path.join(os.path.dirname(__file__), configuration['CustomDataFile'])

        # npy file: floor_number, time_interval, expected number of passengers showing at the 1 floor, 
        #           prob that passengers going from 1 to 1 floor, prob that passengers going from 1 to 2 floor, prob of 1 to 3, prob of 1 to 4...
        self._pedestrian_flow = np.load(self._data_file)    # (288, 112)
        # data format: time, floor_in_flow, floor_out_flow
        self._data_len = self._pedestrian_flow.shape[0]
        self._floor_number = int(self._pedestrian_flow[0][0])   # 10 in this case
        self._pedestrian_flow = self._pedestrian_flow[:, 1:]
        self._in_density = np.zeros([self._data_len, self._floor_number], dtype = 'float32')
        # probability for passengers going from one floor to another floor
        self._out_prob = np.zeros([self._data_len, self._floor_number, self._floor_number], dtype = 'float32')

        assert (self._pedestrian_flow.shape[1] == 1 + self._floor_number * (self._floor_number + 1)), \
            "The column of the dataset file do not match the mansion, %d and %d"%(
                self._pedestrian_flow.shape[1], 1 + self._floor_number * (self._floor_number + 1)
                )
        assert (self._pedestrian_flow[-1][0] < 86400), \
            "The time of the day must < 86400 sec"
        assert (self._pedestrian_flow[0][0] <= 0.0), \
            "The start time of the day must <= 0.0 sec"

        print ("waiting for loading the environment data")
        for i in range(self._data_len):
            if(i < self._data_len - 1):
                tmp_val = self._pedestrian_flow[i + 1][0] - self._pedestrian_flow[i][0]
            else:
                tmp_val = 86400 - self._pedestrian_flow[i][0]
            assert(tmp_val > 0.0), "The time interval must be above zero"
            for j in range(self._floor_number):
                self._in_density[i][j] = 1.0 / tmp_val * self._pedestrian_flow[i][j * (self._floor_number + 1) + 1]
                self._out_prob[i][j] =  self._pedestrian_flow[i][(j * (self._floor_number + 1) + 2) : ((j + 1) * (self._floor_number + 1) + 1)]
            
        self._cur_time_index = 0
        self._cur_id = 0

    def link_mansion(self, mansion_config):
        self._config = mansion_config
        self._last_generate_time = self._config.raw_time

        assert (self._floor_number == self._config.number_of_floors), \
            "The dimension of the data file does not match the floor number, %d and %d"%(
                self._floor_number, self._config.number_of_floors
                ) 

    def _weight_generator(self):
        MIN_WEIGHT = 20
        MAX_WEIGHT = 100
        weight = random.normalvariate(50, 10)
        while weight < MIN_WEIGHT or weight > MAX_WEIGHT:
            weight = random.normalvariate(50, 10)
        return weight

    def _binary_search(self, beg, end, res_time):
        if(beg >= self._data_len - 1):
            return beg
        if(not self._pedestrian_flow[beg + 1][0] < res_time):
            return beg
        if(not self._pedestrian_flow[end][0] > res_time):
            return end
        search_idx = (beg + end) // 2
        if(self._pedestrian_flow[search_idx][0] < res_time):
          return self._binary_search(search_idx, end - 1, res_time)
        else:
          return self._binary_search(beg + 1, search_idx, res_time)

    def _check_time_index(self, time):
        res_time = time % 86400
        if(self._cur_time_index + 1 < self._data_len):
            if(self._pedestrian_flow[self._cur_time_index + 1][0] < res_time):
                self._cur_time_index = self._binary_search(self._cur_time_index + 1, self._data_len - 1, res_time)
        if(self._pedestrian_flow[self._cur_time_index][0] > res_time):
            self._cur_time_index = self._binary_search(0, self._cur_time_index, res_time)

    def generate_person(self):
        '''
        Generate Pedestrian Flow Patterns According to Distributions
        Args:
          None
        Returns:
          List of Random Persons
        '''
        ret_persons = []
        cur_time = self._config.raw_time
        time_interval = cur_time - self._last_generate_time
        self._check_time_index(int(cur_time))
        tmp_in_lambda = self._in_density[self._cur_time_index] * time_interval
        flow_in_person = np.random.poisson(tmp_in_lambda, size = tmp_in_lambda.shape)
        for i in range(self._floor_number):
            if(flow_in_person[i] > 0):
                sample_prob = self._out_prob[self._cur_time_index][i] / (1.0e-5 + self._out_prob[self._cur_time_index][i].sum())
                sample_out_floor = np.random.multinomial(flow_in_person[i], sample_prob)
                for j in range(len(sample_out_floor)):
                    for _ in range(sample_out_floor[j]):
                        ret_persons.append(PersonType(
                        self._cur_id,
                        self._weight_generator(), 
                        i + 1, 
                        j + 1,
                        self._config.raw_time))
                    self._cur_id += 1

        self._last_generate_time = self._config.raw_time

        return ret_persons
