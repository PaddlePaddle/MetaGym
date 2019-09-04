# custom_generators.py
#
# Generating pedestrian flow from npy file
#
# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License.#

import os
import sys
import random
import numpy as np
from rlschool.liftsim.environment.mansion.utils import EPSILON
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.person_generators.person_generator import PersonGeneratorBase


def resample_solver(in_flow_vec, out_flow_vec, f_lambda, max_iteration):
    """
    Given In/Out flow of the floor, it will inference the pedestrian flow in each direction
    """
    shape = np.shape(in_flow_vec)
    shape_2 = np.shape(out_flow_vec)
    assert len(shape) == 1 and shape == shape_2
    n = shape[0]
    row = 2 * n
    col = n * (n - 1)

    def transfer(i, j):
        #transfer flow from i to j to a index
        if(i > j):
            return i * (n - 1) + j
        elif(i < j):
            return i * (n - 1) + j - 1
        else:
            return None

    A = np.zeros(shape=[row, col], dtype = 'float32')
    res_b = np.zeros(shape=[2 * n], dtype = 'float32')
    for i in range(n):
        for j in range(n):
            #The flow from every floor j to floor i
            if(i != j):
                A[i, transfer(j, i)] = in_flow_vec[j]
                #The flow out of every floor i
                A[i + n, transfer(i, j)] = 1.0
        res_b[i] = out_flow_vec[i]
        res_b[i + n] = 1.0

    error = np.linalg.norm(res_b)
    M = np.matmul(A.transpose(), A) + f_lambda * np.eye(col, dtype = 'float32')
    inv_M = np.linalg.inv(M)

    res = np.zeros(shape = [col], dtype='float32')
    iteration = 0
    while(iteration < max_iteration and error > 1.0e-5):
        y = np.matmul(A.transpose(), res_b)
        tmp_res = np.matmul(inv_M, y)

        np.clip(res, 0, None)
        res_b = res_b - np.matmul(A, tmp_res)
        error = np.linalg.norm(res_b)
        res += tmp_res
        iteration += 1
        print(error)

    fin_res = np.zeros(shape=[n, n],dtype='float32')
    for i in range(n):
        for j in range(n):
            if(i != j):
                fin_res[i][j] = res[transfer(i, j)]

    for i in range(n):
        fin_res[i] /= np.sum(fin_res[i])

    #print (np.matmul(fin_res.transpose(), in_flow_vec) - out_flow_vec)
    return fin_res

  

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

        self._pedestrian_flow = np.load(self._data_file)
        #data format: time, floor_in_flow, floor_out_flow
        self._data_len = self._pedestrian_flow.shape[0]
        self._floor_number = (self._pedestrian_flow.shape[1] - 1) // 2
        self._in_density = np.zeros([self._data_len, self._floor_number], dtype = 'float32')
        self._out_density = np.zeros([self._data_len, self._floor_number], dtype = 'float32')
        self._out_prob = np.zeros([self._data_len, self._floor_number], dtype = 'float32')
        assert (self._pedestrian_flow.shape[1] % 2 == 1), \
            "The column of the npy file must be singular"
        assert (self._pedestrian_flow[-1][0] < 86400), \
            "The time of the day must < 86400 sec"
        assert (self._pedestrian_flow[0][0] < 0.0), \
            "The start time of the day must < 0.0 sec"
        for i in range(self._data_len):
            if(i < self._data_len - 1):
                tmp_val = self._pedestrian_flow[i + 1][0] - self._pedestrian_flow[i][0]
            else:
                tmp_val = 86400 - self._pedestrian_flow[i][0]
            assert(tmp_val > 0.0), "The time interval must be above zero"
            self._in_density[i] = 1.0 / tmp_val * self._pedestrian_flow[i][1:(self._floor_number+1)]
            self._out_density[i] = 1.0 / tmp_val * self._pedestrian_flow[i][(self._floor_number+1):(2*self._floor_number+1)]
            self._out_prob[i] = resample_solver(self._in_density[i], self._out_density[i], 1.0e-3, 10)
            
        self._cur_time_index = 0
        self._cur_id = 0

    def link_mansion(self, mansion_config):
        self._config = mansion_config
        self._last_generate_time = self._config.raw_time

        assert (self._floor_number == self._config.number_of_floors), \
            "The dimension of the data file must match the floor number" 

    def _weight_generator(self):
        return random.normalvariate(50, 10)

    def _binary_search(self, beg, end, res_time):
        if(beg >= self._data_len - 1):
            return beg
        if(not self._pedestrian_flow[beg + 1][0] < res_time):
            return beg + 1
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
            sample_out_floor = np.random.multinomial(flow_in_person[i], self._out_prob[i], size = 1)
            for j in sample_out_floor:
                for _ in sample_out_floor:
                    ret_persons.append(PersonType(
                      self._cur_id,
                      self._weight_generator(), 
                      i + 1, 
                      j + 1,
                      self._config.raw_time))
                    self._cur_id += 1

        self._last_generate_time = self._config.raw_time
        return ret_persons
