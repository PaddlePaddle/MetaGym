# basic_generators.py
# Author: Fan Wang (wang.fan@baidu.com)
#
# A Proxy for generating different persons

import sys
import random
from six import integer_types
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.person_generators.uniform_generator import UniformPersonGenerator
from rlschool.liftsim.environment.mansion.person_generators.office_generator import OfficePersonGenerator


def PersonGenerator(gen_type):
    if(gen_type == "UNIFORM"):
        return UniformPersonGenerator()
    elif(gen_type == "OFFICE"):
        return OfficePersonGenerator()
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
