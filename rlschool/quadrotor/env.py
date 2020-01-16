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

# Extension module
import quadrotorsim

NO_DISPLAY = False
try:
    from rlschool.quadrotor.render import RenderWindow
except Exception as e:
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
        obs_as_dict (bool): whether to return observation as dict.
    """
    def __init__(self,
                 dt=0.01,
                 nt=1000,
                 seed=0,
                 task='no_collision',
                 map_file=None,
                 simulator_conf=None,
                 obs_as_dict=False,
                 **kwargs):
        assert task in ['velocity_control', 'no_collision'], \
            'Invalid task setting'
        if simulator_conf is None:
            simulator_conf = os.path.join(os.path.dirname(__file__),
                                          'quadrotorsim', 'config.xml')
        assert os.path.exists(simulator_conf), \
            'Simulator config xml does not exist'

        self.dt = dt
        self.nt = nt
        self.ct = 0
        self.task = task
        self.obs_as_dict = obs_as_dict
        self.simulator = quadrotorsim.Simulator()

        cfg_dict = self.simulator.get_config(simulator_conf)
        self.action_space = namedtuple(
            'action_space', ['shape', 'high', 'low', 'sample'])
        self.action_space.shape = [4]
        self.action_space.high = [cfg_dict['action_space_high']] * 4
        self.action_space.low = [cfg_dict['action_space_low']] * 4
        self.action_space.sample = Quadrotor.random_action(
            cfg_dict['action_space_low'], cfg_dict['action_space_high'], 4)

        # self.position_keys = ['x', 'y', 'z']
        self.position_keys = ['z']
        self.global_velocity_keys = ['g_v_x', 'g_v_y', 'g_v_z']
        self.body_velocity_keys = ['b_v_x', 'b_v_y', 'b_v_z']
        self.angular_velocity_keys = ['w_x', 'w_y', 'w_z']
        self.flight_pose_keys = ['pitch', 'roll', 'yaw']
        self.accelerator_keys = ['acc_x', 'acc_y', 'acc_z']
        self.gyroscope_keys = ['gyro_x', 'gyro_y', 'gyro_z']
        self.extra_info_keys = ['power']
        self.task_velocity_control_keys = \
            ['next_target_g_v_x', 'next_target_g_v_y', 'next_target_g_v_z']

        obs_dim = len(self.position_keys) + len(self.global_velocity_keys) + \
            len(self.body_velocity_keys) + len(self.angular_velocity_keys) + \
            len(self.flight_pose_keys) + len(self.accelerator_keys) + \
            len(self.gyroscope_keys) + len(self.extra_info_keys)
        if self.task == 'velocity_control':
            obs_dim += len(self.task_velocity_control_keys)
        self.observation_space = namedtuple('observation_space', ['shape'])
        self.observation_space.shape = [obs_dim]

        self.state = {}
        self.viewer = None
        self.x_offset = self.y_offset = self.z_offset = 0

        if self.task == 'velocity_control':
            self.velocity_targets = \
                self.simulator.define_velocity_control_task(
                    dt, nt, seed)
        elif self.task == 'no_collision':
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

        if self.obs_as_dict:
            return self.state
        else:
            return self._convert_state_to_ndarray()

    def step(self, action):
        self.ct += 1
        cmd = np.asarray(action, np.float32)
        self.simulator.step(cmd.tolist(), self.dt)
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()

        old_pos = [self.state['x'], self.state['y'], self.state['z']]
        self._update_state(sensor_dict, state_dict)
        new_pos = [self.state['x'], self.state['y'], self.state['z']]
        if self.task == 'no_collision':
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

        if self.obs_as_dict:
            return self.state, reward, reset
        else:
            return self._convert_state_to_ndarray(), reward, reset

    def render(self):
        if self.viewer is None:
            if NO_DISPLAY:
                raise RuntimeError('[Error] Cannot connect to display screen.')
            self.viewer = RenderWindow(task=self.task)

        if 'x' not in self.state:
            # It's null state
            raise Exception('You are trying to render before calling reset()')

        if self.task == 'velocity_control':
            self.viewer.view(
                self.state, self.dt,
                expected_velocity=self.velocity_targets[self.ct-1])
        else:
            self.viewer.view(self.state, self.dt)

    def close(self):
        del self.simulator

    def _convert_state_to_ndarray(self):
        keys_order = self.position_keys + self.global_velocity_keys + \
            self.body_velocity_keys + self.angular_velocity_keys + \
            self.flight_pose_keys + self.accelerator_keys + \
            self.gyroscope_keys + self.extra_info_keys

        if self.task == 'velocity_control':
            keys_order.extend(self.task_velocity_control_keys)

        ndarray = np.array([self.state[k] for k in keys_order])
        return ndarray

    def _get_reward(self, collision=False, velocity_target=(0.0, 0.0, 0.0)):
        """
        Reward function setting for different tasks.

        The default penalty is the cost of energy. In addition,
        for `no_collision` task, a strong penalty is added for
        collision, otherwise get +1 reward; for `velocity_control`
        task, an extra penalty for velocity difference is added.
        """
        reward = - self.dt * self.state['power']
        if self.task == 'no_collision':
            if collision:
                reward -= 10.0
            else:
                reward += 1.
        elif self.task == 'velocity_control':
            diff = self._get_velocity_diff(velocity_target)
            reward -= diff * 0.001

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
        state['x'] = state['x'] + self.x_offset
        state['y'] = state['y'] + self.y_offset
        state['z'] = state['z'] + self.z_offset

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
        diff = abs(vt_x - self.state['g_v_x']) + \
            abs(vt_y - self.state['g_v_y']) + \
            abs(vt_z - self.state['g_v_z'])
        return diff

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
        def sample():
            act = np.random.random_sample((dim,))
            return (high - low) * act + low
        return sample


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        task = 'no_collision'
    else:
        task = sys.argv[1]
    env = Quadrotor(task=task, nt=1000)
    import ipdb; ipdb.set_trace()
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
    env.close()
