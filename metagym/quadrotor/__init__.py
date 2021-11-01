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

from gym.envs.registration import register
from metagym.quadrotor.env import Quadrotor

__all__ = ['Quadrotor']

register(
    id='quadrotor-v0',
    entry_point='metagym.quadrotor:Quadrotor',
    kwargs={
        "dt":0.01,
        "nt":1000,
        "seed":0,
        "task":'no_collision',
        "map_file":None,
        "simulator_conf":None,
        "healthy_reward":1.0,
    }
)
