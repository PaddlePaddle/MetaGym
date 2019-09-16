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

# Author: Fan Wang (wang.fan@baidu.com)
#
# A Proxy for generating different persons

import sys
import random
import numpy as np
from six import integer_types
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.person_generators.uniform_generator import UniformPersonGenerator
from rlschool.liftsim.environment.mansion.person_generators.custom_generator import CustomGenerator


def PersonGenerator(gen_type):
    if(gen_type == "UNIFORM"):
        return UniformPersonGenerator()
    elif(gen_type == "CUSTOM"):
        return CustomGenerator()
    else:
        raise RuntimeError("No such generator type: %s" % gen_type)


def set_seed(seed):
    if seed is not None and not (
        isinstance(
            seed,
            integer_types) and 0 <= seed):
        raise Exception(
            'Seed must be a non-negative integer or omitted, not {}'.format(seed))
    else:
        random.seed(seed)
        np.random.seed(seed)
