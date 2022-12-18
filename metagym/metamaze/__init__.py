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
from metagym.metamaze.envs import MetaMazeContinuous3D
from metagym.metamaze.envs import MetaMazeDiscrete3D
from metagym.metamaze.envs import MetaMaze2D
from metagym.metamaze.envs import MazeTaskSampler

register(
    id='meta-maze-continuous-3D-v0',
    entry_point='metagym.metamaze:MetaMazeContinuous3D',
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 5000,
        "task_type": "SURVIVAL"
    }
)

register(
    id='meta-maze-discrete-3D-v0',
    entry_point='metagym.metamaze:MetaMazeDiscrete3D',
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 200,
        "task_type": "SURVIVAL"
    }
)

register(
    id='meta-maze-2D-v0',
    entry_point='metagym.metamaze:MetaMaze2D',
    kwargs={
        "enable_render": True,
        "max_steps": 200,
        "view_grid": 1,
        "task_type": "SURVIVAL"
    }
)
