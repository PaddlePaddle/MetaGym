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
import paddle.fluid as fluid
import parl.layers as layers
import numpy as np
from parl.framework.model_base import Model
from parl.framework.agent_base import Agent
from parl.utils import get_gpu_count


class RLDispatcherModel(Model):
    def __init__(self, act_dim):
        self._act_dim = act_dim
        self._fc_1 = layers.fc(size=512, act='relu')
        self._fc_2 = layers.fc(size=256, act='relu')
        self._fc_3 = layers.fc(size=128, act='tanh')
        self._output = layers.fc(size=act_dim)

    def value(self, obs):
        self._h_1 = self._fc_1(obs)
        self._h_2 = self._fc_2(self._h_1)
        self._h_3 = self._fc_3(self._h_2)
        self._pred = self._output(self._h_3)
        return self._pred


class ElevatorAgent(Agent):
    def __init__(self, algorithm, obs_dim, action_dim):
        self._action_dim = action_dim
        self._obs_dim = obs_dim
        self._update_target_steps = 1000
        self._global_step = 0
        super(ElevatorAgent, self).__init__(algorithm)

        use_cuda = True if self.gpu_id >= 0 else False
        if self.gpu_id >= 0:
            assert get_gpu_count() == 1, 'Only support training in single GPU,\
                    Please set environment variable: `export CUDA_VISIBLE_DEVICES=[GPU_ID_YOU_WANT_TO_USE]` .'

        else:
            os.environ['CPU_NUM'] = str(1)
            # cpu_num = os.environ.get('CPU_NUM')
            # assert cpu_num is not None and cpu_num == '1', 'Only support training in single CPU,\
            #         Please set environment variable:  `export CPU_NUM=1`.'

        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        exec_strategy.num_iteration_per_drop_scope = 10
        build_strategy = fluid.BuildStrategy()
        build_strategy.remove_unnecessary_lock = False

        self.learn_pe = fluid.ParallelExecutor(
            use_cuda=use_cuda,
            main_program=self._learn_program,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy,
            )

    def build_program(self):
        self._pred_program = fluid.Program()
        self._learn_program = fluid.Program()

        with fluid.program_guard(self._pred_program):
            obs = layers.data(
                name='obs',
                shape=[self._obs_dim],
                dtype='float32'
            )
            self._value = self.alg.define_predict(obs)

        with fluid.program_guard(self._learn_program):
            obs = layers.data(
                name='obs',
                shape=[self._obs_dim],
                dtype='float32'
            )
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs',
                shape=[self._obs_dim],
                dtype='float32'
            )
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self._cost = self.alg.define_learn(
                obs, action, reward, next_obs, terminal)
    
    def predict(self, obs):
        #obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self._pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self._value])  # [0]
        #pred_Q = np.squeeze(pred_Q, axis=0)
        return pred_Q[0]

    def learn(self, obs, act, reward, next_obs, terminal):
        self._global_step += 1
        if self._global_step % self._update_target_steps == 0:
            self.alg.sync_target(self.gpu_id)

        #act = np.expand_dims(act, -1)
        #print("observation:", self._obs_dim, self._action_dim, np.shape(obs), np.shape(act), np.shape(reward), np.shape(next_obs), np.shape(terminal))
        #reward = np.clip(reward, -1, 1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.learn_pe.run(
            feed=feed, fetch_list=[self._cost.name])[0]
        return cost
