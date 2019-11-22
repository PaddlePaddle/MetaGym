#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import time
import numpy as np
from math import floor, ceil

# Extension module
import quadrotorsim

NO_DISPLAY = False
from rlschool.quadrotor.render import RenderWindow
try:
    from rlschool.quadrotor.render import RenderWindow
except Exception as e:
    NO_DISPLAY = True


m = 0.50
U_T = 0.025
I_1 = 1.35e-2
I_2 = 1.35e-2
I_3 = 2.40e-2
L = 0.25
C = 2.50

U_to_T = np.asarray(
    [[U_T / m, U_T / m, U_T / m, U_T / m],
     [-L * U_T / I_1, -L * U_T / I_1, L * U_T / I_1, L * U_T / I_1],
     [-L * U_T / I_2, L * U_T / I_2, L * U_T / I_2, -L * U_T / I_2],
     [C * U_T, -C * U_T, C * U_T, -C * U_T]], 'float32')
T_to_U = np.linalg.inv(U_to_T)


class Quadrotor(object):
    def __init__(self, map_file=None, dt=0.1):
        sim_conf = os.path.join(os.path.dirname(__file__),
                                'quadrotorsim', 'config.xml')
        self.dt = dt
        self.map_matrix = Quadrotor.load_map(map_file)
        self.simulator = quadrotorsim.Simulator()
        self.simulator.get_config(sim_conf)
        self.state = {}
        self.viewer = None

        # Only for single quadrotor, also mark its start position
        y_offsets, x_offsets = np.where(self.map_matrix == -1)
        assert len(y_offsets) == 1
        self.y_offset = y_offsets[0]
        self.x_offset = x_offsets[0]
        self.z_offset = 5.  # TODO: setup a better init height
        self.map_matrix[self.y_offset, self.x_offset] = 0

    def reset(self):
        self.simulator.reset()
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()
        self._update_state(sensor_dict, state_dict)

        return self.state

    def step(self, action):
        cmd = np.asarray(action, np.float32)
        act = np.matmul(T_to_U, cmd)
        self.simulator.step(act.tolist(), self.dt)
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()

        old_pos = [self.state['x'], self.state['y'], self.state['z']]
        self._update_state(sensor_dict, state_dict)
        new_pos = [self.state['x'], self.state['y'], self.state['z']]
        if not self._check_collision(old_pos, new_pos):
            reset = False
        else:
            reset = True

        reward = self._get_reward(collision=reset)

        return self.state, reward, reset

    def render(self):
        if self.viewer is None:
            if NO_DISPLAY:
                raise RuntimeError('[Error] Cannot connect to display screen.')
            self.viewer = RenderWindow()

        if 'x' not in self.state:
            # It's null state
            raise Exception('You are trying to render before calling reset()')

        self.viewer.view(self.state, self.dt)
        time.sleep(self.dt)

    def close(self):
        del self.simulator
        del self.map_matrix

    def _get_reward(self, collision=False):
        # TODO: setup reward function following spoken commands
        return 1.

    def _check_collision(self, old_pos, new_pos):
        # TODO: update to consider the body size of the quadrotor
        min_max = lambda x, y, i: \
            (int(floor(min(x[i], y[i]))), int(ceil(max(x[i], y[i]))))
        x_min, x_max = min_max(old_pos, new_pos, 0)
        y_min, y_max = min_max(old_pos, new_pos, 1)
        z_min, z_max = min_max(old_pos, new_pos, 2)

        taken_pos = self.map_matrix[y_min:y_max+1, x_min:x_max+1]
        if z_min < np.any(taken_pos) or z_max < np.any(taken_pos):
            return True
        else:
            return False

    def _update_state(self, sensor, state):
        state['x'] = state['x'] + self.x_offset
        state['y'] = state['y'] + self.y_offset
        state['z'] = state['z'] + self.z_offset

        for k, v in sensor.items():
            self.state[k] = v

        for k, v in state.items():
            self.state[k] = v

    @staticmethod
    def load_map(map_file):
        if map_file is None:
            flatten_map = np.zeros([100, 100], dtype=np.int32)
            flatten_map[50, 50] = -1
            return flatten_map

        map_lists = []
        with open(map_file, 'r') as f:
            for line in f.readlines():
                map_lists.append([int(i) for i in line.split(' ')])

        return np.array(map_lists)


if __name__ == '__main__':
    env = Quadrotor()
    env.reset()
    env.render()
    reset = False
    step = 1
    while not reset:
        action = np.array([2., 2., 1., 1.], dtype=np.float32)
        # action = np.array([1., 0., 0., 0.], dtype=np.float32)
        state, reward, reset = env.step(action)
        env.render()
        print('---------- step %s ----------' % step)
        print('state:', state)
        print('reward:', reward)
        step += 1
