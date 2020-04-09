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

import unittest
from rlschool import make_env


class TestQuadrotorEnv(unittest.TestCase):
    def test_no_collision_task(self):
        env = make_env('Quadrotor', task='no_collision')
        state = env.reset()
        act = env.action_space.sample()
        reset = False
        while not reset:
            state, reward, reset, info = env.step(act)
            act = env.action_space.sample()

    def test_velocity_control_task(self):
        env = make_env('Quadrotor', task='velocity_control')
        state = env.reset()
        reset = False
        step = 0
        while not reset:
            state, reward, reset, info = env.step([1., 1., 1., 1.])
            self.assertTrue('next_target_g_v_x' in info)
            step += 1

        self.assertEqual(step, env.nt)

    def test_hovering_control_task(self):
        env = make_env('Quadrotor', task='hovering_control')
        state = env.reset()
        act = env.action_space.sample()
        reset = False
        while not reset:
            state, reward, reset, info = env.step(act)
            act = env.action_space.sample()


if __name__ == '__main__':
    unittest.main()
