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

from gym.envs.registration import register
from metagym.metalocomotion.envs.meta_ants.meta_ant_env import MetaAntEnv
from metagym.metalocomotion.envs.meta_humanoids.meta_humanoids_env import MetaHumanoidEnv

register(
    id='meta-ant-v0',
    entry_point='metagym.metalocomotion.envs.meta_ants.meta_ant_env:MetaAntEnv',
    kwargs={"frame_skip": 4,
        "time_step": 0.005,
        "enable_render": False,
        "max_steps": 2000,
    }
)

register(
    id='meta-humanoid-v0',
    entry_point='metagym.metalocomotion.envs.meta_humanoids.meta_humanoids_env:MetaHumanoidEnv',
    kwargs={"frame_skip": 4,
        "time_step": 0.005,
        "enable_render": False,
        "max_steps": 2000,
    }
)
