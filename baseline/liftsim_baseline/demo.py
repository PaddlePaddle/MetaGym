# -*- coding: UTF-8 -*-
##########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
##########################################################################

"""

A running demo of elevators

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

from rlschool.liftsim.environment.env import LiftSim
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.utils import ElevatorState, MansionState
from rlschool.liftsim.environment.mansion.mansion_manager import MansionManager
from baseline.wrapper import Wrapper, ActionWrapper, ObservationWrapper
from rl_benchmark.dispatcher import RL_dispatcher
import sys
import argparse

# run main program with args
def run_main(args):

    parser = argparse.ArgumentParser(description='demo configuration')
    parser.add_argument('--iterations', type=int, default=100000000,
                            help='total number of iterations')
    args = parser.parse_args(args)
    print('iterations:', args.iterations)

    mansion_env = LiftSim()
    mansion_env.seed(2019)

    mansion_env = Wrapper(mansion_env)
    mansion_env = ActionWrapper(mansion_env)
    mansion_env = ObservationWrapper(mansion_env)

    dispatcher = RL_dispatcher(mansion_env, args.iterations)
    dispatcher.run_episode()

    return 0


if __name__ == "__main__":
    run_main(sys.argv[1:])
