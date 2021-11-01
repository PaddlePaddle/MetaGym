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
from metagym.navigator2d.navigator import Navigator2D
from metagym.navigator2d.wheeled_robot import WheeledRobot

register(
    id='navigator-wr-2D-v0',
    entry_point='metagym.navigator2d.navigator:Navigator2D',
    kwargs={
        "max_steps": 100,
        "robot_class": WheeledRobot,
        "signal_noise": 0.20,
        "enable_render": True
    }
)
