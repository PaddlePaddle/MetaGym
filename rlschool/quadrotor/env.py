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
import numpy as np
from math import floor, ceil
from collections import namedtuple

from rlschool.quadrotor.quadrotorsim import QuadrotorSim

NO_DISPLAY = False
try:
    from rlschool.quadrotor.render import RenderWindow
except Exception:
    NO_DISPLAY = True


class Quadrotor(object):
    """
    Quadrotor environment.

    Args:
        dt (float): duration of single step (in seconds).
        nt (int): number of steps of single episode if no collision
            occurs.
        seed (int): seed to generate target velocity trajectory.
        task (str): name of the task setting. Currently, support
            `no_collision` and `velocity_control`.
        map_file (None|str): path to txt map config file, default
            map is a 100x100 flatten floor.
        simulator_conf (None|str): path to simulator config xml file.
    """
    def __init__(self,
                 dt=0.01,
                 nt=1000,
                 seed=0,
                 task='no_collision',
                 map_file=None,
                 simulator_conf=None,
                 healthy_reward=10.0,
                 **kwargs):
        # TODO: other possible tasks: precision_landing
        assert task in ['velocity_control', 'no_collision',
                        'hovering_control'], 'Invalid task setting'
        if simulator_conf is None:
            simulator_conf = os.path.join(os.path.dirname(__file__),
                                          'config.json')
        assert os.path.exists(simulator_conf), \
            'Simulator config file does not exist'

        self.dt = dt
        self.nt = nt
        self.ct = 0
        self.task = task
        self.healthy_reward = healthy_reward
        self.simulator = QuadrotorSim()

        cfg_dict = self.simulator.get_config(simulator_conf)
        self.valid_range = cfg_dict['range']
        self.action_space = namedtuple(
            'action_space', ['shape', 'high', 'low', 'sample'])
        self.action_space.shape = [4]
        self.action_space.high = [cfg_dict['action_space_high']] * 4
        self.action_space.low = [cfg_dict['action_space_low']] * 4
        self.action_space.sample = Quadrotor.random_action(
            cfg_dict['action_space_low'], cfg_dict['action_space_high'], 4)

        self.body_velocity_keys = ['b_v_x', 'b_v_y', 'b_v_z']
        self.accelerator_keys = ['acc_x', 'acc_y', 'acc_z']
        self.gyroscope_keys = ['gyro_x', 'gyro_y', 'gyro_z']
        self.flight_pose_keys = ['pitch', 'roll', 'yaw']
        self.barometer_keys = ['z']
        self.task_velocity_control_keys = \
            ['next_target_g_v_x', 'next_target_g_v_y', 'next_target_g_v_z']

        obs_dim = len(self.body_velocity_keys) + len(self.accelerator_keys) + \
            len(self.gyroscope_keys) + len(self.flight_pose_keys) + \
            len(self.barometer_keys)
        if self.task == 'velocity_control':
            obs_dim += len(self.task_velocity_control_keys)
        self.observation_space = namedtuple('observation_space', ['shape'])
        self.observation_space.shape = [obs_dim]

        self.state = {}
        self.viewer = None
        self.x_offset = self.y_offset = self.z_offset = 0
        self.pos_0 = np.array([0.0] * 3).astype(np.float32)

        if self.task == 'velocity_control':
            self.velocity_targets = \
                self.simulator.define_velocity_control_task(
                    dt, nt, seed)
        elif self.task in ['no_collision', 'hovering_control']:
            self.map_matrix = Quadrotor.load_map(map_file)

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

        # Mark the initial position
        self.pos_0 = np.copy(self.simulator.global_position)

        return self._convert_state_to_ndarray()

    def step(self, action):
        self.ct += 1
        cmd = np.asarray(action, np.float32)
        self.simulator.step(cmd.tolist(), self.dt)
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()

        old_pos = [self.simulator.global_position[0] + self.x_offset,
                   self.simulator.global_position[1] + self.y_offset,
                   self.simulator.global_position[2] + self.z_offset]
        self._update_state(sensor_dict, state_dict)
        new_pos = [self.simulator.global_position[0] + self.x_offset,
                   self.simulator.global_position[1] + self.y_offset,
                   self.simulator.global_position[2] + self.z_offset]
        if self.task in ['no_collision', 'hovering_control']:
            is_collision = self._check_collision(old_pos, new_pos)
            reward = self._get_reward(collision=is_collision)
            reset = False
            if is_collision:
                reset = True
                self.ct = 0
        elif self.task == 'velocity_control':
            reset = False
            velocity_target = self.velocity_targets[self.ct - 1]
            reward = self._get_reward(velocity_target=velocity_target)

        if self.ct == self.nt:
            reset = True
            self.ct = 0

        info = {k: self.state[k] for k in self.state.keys()}
        info['z'] += self.z_offset
        return self._convert_state_to_ndarray(), reward, reset, info

    def render(self):
        if self.viewer is None:
            if NO_DISPLAY:
                raise RuntimeError('[Error] Cannot connect to display screen.')
            self.viewer = RenderWindow(task=self.task,
                                       x_offset=self.x_offset,
                                       y_offset=self.y_offset,
                                       z_offset=self.z_offset)

        if 'z' not in self.state:
            # It's null state
            raise Exception('You are trying to render before calling reset()')

        state = self._get_state_for_viewer()
        if self.task == 'velocity_control':
            self.viewer.view(
                state, self.dt,
                expected_velocity=self.velocity_targets[self.ct-1])
        else:
            self.viewer.view(state, self.dt)

    def close(self):
        del self.simulator

    def _convert_state_to_ndarray(self):
        keys_order = self.body_velocity_keys + self.accelerator_keys + \
            self.gyroscope_keys + self.flight_pose_keys + \
            self.barometer_keys

        if self.task == 'velocity_control':
            keys_order.extend(self.task_velocity_control_keys)

        ndarray = []
        for k in keys_order:
            if k == 'z':
                ndarray.append(self.state[k] + self.z_offset)
            else:
                ndarray.append(self.state[k])

        ndarray = np.array(ndarray, dtype=np.float32)
        return ndarray

    def _get_reward(self, collision=False, velocity_target=(0.0, 0.0, 0.0)):
        """
        Reward function setting for different tasks.
        """
        # Make sure energy cost always smaller than healthy reward,
        # to encourage longer running
        reward = - min(self.dt * self.simulator.power, self.healthy_reward)
        if self.task == 'no_collision':
            task_reward = 0.0 if collision else self.healthy_reward
            reward += task_reward
        elif self.task == 'velocity_control':
            task_reward = -0.001 * self._get_velocity_diff(velocity_target)
            reward += task_reward
        elif self.task == 'hovering_control':
            task_reward = 0.0 if collision else self.healthy_reward

            velocity_norm = np.linalg.norm(
                self.simulator.global_velocity)
            angular_velocity_norm = np.linalg.norm(
                self.simulator.body_angular_velocity)

            # TODO: you may find better ones!
            alpha, beta = 1.0, 1.0
            hovering_range, in_range_r, out_range_r = 0.5, 10, -20

            task_reward -= alpha * velocity_norm + beta * angular_velocity_norm

            z_move = abs(self.pos_0[2] - self.state['z'])
            if z_move < hovering_range:
                task_reward += in_range_r
            else:
                task_reward += max(out_range_r, hovering_range - z_move)

            reward += task_reward

        return reward

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
        for k, v in sensor.items():
            self.state[k] = v

        for k, v in state.items():
            self.state[k] = v

        if self.task == 'velocity_control':
            t = min(self.ct, self.nt-1)
            next_velocity_target = self.velocity_targets[t]
            self.state['next_target_g_v_x'] = next_velocity_target[0]
            self.state['next_target_g_v_y'] = next_velocity_target[1]
            self.state['next_target_g_v_z'] = next_velocity_target[2]

    def _get_velocity_diff(self, velocity_target):
        vt_x, vt_y, vt_z = velocity_target
        diff = abs(vt_x - self.state['b_v_x']) + \
            abs(vt_y - self.state['b_v_y']) + \
            abs(vt_z - self.state['b_v_z'])
        return diff

    def _get_state_for_viewer(self):
        state = {k: v for k, v in self.state.items()}
        state['x'] = self.simulator.global_position[0]
        state['y'] = self.simulator.global_position[1]
        state['z'] = self.simulator.global_position[2]
        state['g_v_x'] = self.simulator.global_velocity[0]
        state['g_v_y'] = self.simulator.global_velocity[1]
        state['g_v_z'] = self.simulator.global_velocity[2]
        return state

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

    @staticmethod
    def random_action(low, high, dim):
        @staticmethod
        def sample():
            act = np.random.random_sample((dim,))
            return (high - low) * act + low
        return sample


if __name__ == '__main__':
    import sys
    import time
    if len(sys.argv) == 1:
        task = 'no_collision'
    else:
        task = sys.argv[1]
    env = Quadrotor(task=task, nt=1000)
    env.reset()
    env.render()
    reset = False
    step = 1
    total_reward = 0
    ts = time.time()
    while not reset:
        action = np.array([5., 5., 5., 5.], dtype=np.float32)
        state, reward, reset, info = env.step(action)
        total_reward += reward
        env.render()
        print('---------- step %s ----------' % step)
        print('state:', state)
        print('info:', info)
        print('reward:', reward)
        step += 1
    env.close()
    print('total reward: ', total_reward)
    te = time.time()
    print('time cost: ', te - ts)
