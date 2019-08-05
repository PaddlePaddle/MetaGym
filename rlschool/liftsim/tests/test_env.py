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

from rlschool.liftsim.environment.mansion.utils import MansionAttribute, MansionState
from rlschool.liftsim.environment.env import LiftSim

env = LiftSim()
env.seed(1998)
#iteration = env.iterations
state = env.reset()
action = [0,1,0,1,0,1,0,1]
for i in range(100):
    next_state, reward, _, _ = env.step(action)

assert isinstance(env.attribute, MansionAttribute)
assert isinstance(env.state, MansionState)
print(env.statistics)
env.log_debug("This is a debug log")
env.log_notice("This is a notice log")
