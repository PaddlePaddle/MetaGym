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

from rlschool.liftsim.environment.env import LiftSim, Wrapper, ActionWrapper, ObservationWrapper
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.utils import ElevatorState, MansionState
from rlschool.liftsim.environment.mansion.mansion_manager import MansionManager
from rl_benchmark.dispatcher import RL_dispatcher
from rule_benchmark.dispatcher import Rule_dispatcher
import sys
import argparse


def run_mansion_main(mansion_env, policy_handle, iteration):
    mansion_env.reset()
    policy_handle.link_mansion(mansion_env.attribute)
    policy_handle.load_settings()
    i = 0
    acc_reward = 0.0
    while i < iteration:
        i += 1
        mansion_env.render()
        state = mansion_env.state
        action = policy_handle.policy(state)
        _, r, _, _ = mansion_env.step(action)
        output_info = policy_handle.feedback(state, action, r)
        acc_reward += r
        if(isinstance(output_info, dict) and len(output_info) > 0):
            mansion_env.log_notice("%s", output_info)
        if(i % 3600 == 0):
            mansion_env.log_notice(
                "Accumulated Reward: %f, Mansion Status: %s",
                acc_reward, mansion_env.statistics)
            acc_reward = 0.0

# run main program with args
def run_main(args):

    parser = argparse.ArgumentParser(description='demo configuration')
    parser.add_argument('--iterations', type=int, default=100000000,
                            help='total number of iterations')
    parser.add_argument('--controlpolicy', type=str, default='rule_benchmark',
                            help='policy type: rule_benchmark or others')
    args = parser.parse_args(args)
    print('iterations:', args.iterations)
    print('controlpolicy:', args.controlpolicy)

    mansion_env = LiftSim()
    mansion_env.seed(2019)

    if args.controlpolicy == 'rule_benchmark':
        dispatcher = Rule_dispatcher(mansion_env, args.iterations)
        dispatcher.run_dispacher()
    elif args.controlpolicy == 'rl_benchmark':
        mansion_env = Wrapper(mansion_env)
        mansion_env = ActionWrapper(mansion_env)
        mansion_env = ObservationWrapper(mansion_env)

        dispatcher = RL_dispatcher(mansion_env, args.iterations)
        dispatcher.run_episode()

    return 0


if __name__ == "__main__":
    run_main(sys.argv[1:])
