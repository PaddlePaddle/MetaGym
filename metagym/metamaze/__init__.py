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
from metagym.metamaze.envs import MetaMaze3D
from metagym.metamaze.envs import MetaMaze2D

register(
    id='meta-maze-3D-v0',
    entry_point='metagym.metamaze:MetaMaze3D',
    kwargs={"with_guidepost": False,
        "enable_render": True,
        "render_scale": 480,
        "render_godview": True,
        "resolution": (256, 256),
        "max_steps": 1000,
    }
)

register(
    id='meta-maze-2D-v0',
    entry_point='metagym.metamaze:MetaMaze2D',
    kwargs={
        "enable_render": True,
        "render_godview": True,
        "max_steps": 200,
        "view_grid": 1
    }
)
