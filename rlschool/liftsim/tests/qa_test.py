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

"""

qa test for elevators

Authors: likejiao(likejiao@baidu.com)
Date:    2019/06/16 19:30:16
"""

import sys
import time
import copy
import traceback

from rlschool.liftsim.environment.env import LiftSim
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.utils import ElevatorState, MansionState, ElevatorAction
from rlschool.liftsim.environment.mansion.mansion_manager import MansionManager
from rule_benchmark.dispatcher import Rule_dispatcher


fail_flag = False
stop_count = 10


def state_check(state, next_state, action):
    global fail_flag
    global stop_count
    try:
        assert isinstance(state, MansionState)
        # for e in state.ElevatorStates:
        for i in range(len(state.ElevatorStates)):
            ele = copy.deepcopy(state.ElevatorStates[i])
            assert isinstance(ele, ElevatorState)
            next_ele = copy.deepcopy(next_state.ElevatorStates[i])
            assert isinstance(next_ele, ElevatorState)
            act = copy.deepcopy(action[i])
            assert isinstance(act, ElevatorAction)
            # type
            ele_Floor = ele.Floor
            ele_Velocity = ele.Velocity
            ele_LoadWeight = ele.LoadWeight
            next_ele_Floor = next_ele.Floor
            next_ele_Velocity = next_ele.Velocity
            next_ele_LoadWeight = next_ele.LoadWeight

            assert isinstance(ele_Floor, float)
            assert isinstance(ele.MaximumFloor, int)
            assert isinstance(ele_Velocity, float)
            assert isinstance(ele.MaximumSpeed, float)
            assert isinstance(ele.Direction, int)
            assert isinstance(ele.CurrentDispatchTarget, int)
            assert isinstance(ele.DispatchTargetDirection, int)
            assert isinstance(ele_LoadWeight, float)
            assert isinstance(ele.MaximumLoad, int)
            assert isinstance(ele.OverloadedAlarm, float)
            assert isinstance(ele.DoorIsOpening, bool)
            assert isinstance(ele.DoorIsClosing, bool)
            assert isinstance(ele.ReservedTargetFloors, list)
            # change
            ele_Floor = round(ele_Floor, 2)
            ele_Velocity = round(ele_Velocity, 2)
            ele_LoadWeight = round(ele_LoadWeight, 2)
            next_ele_Velocity = round(next_ele_Velocity, 2)
            ele_Velocity = round(ele_Velocity, 2)
            next_ele_LoadWeight = round(next_ele_LoadWeight, 2)
            
            # range
            assert ele_Floor > 0 and ele_Floor <= ele.MaximumFloor
            assert ele_Velocity >= (0 - ele.MaximumSpeed) and ele_Velocity <= ele.MaximumSpeed
            assert ele.Direction in [-1, 0, 1]
            assert ele.CurrentDispatchTarget >= -1 and ele.CurrentDispatchTarget <= ele.MaximumFloor
            assert ele.DispatchTargetDirection in [-1, 1]
            assert ele_LoadWeight >= 0 and ele_LoadWeight <= ele.MaximumLoad
            assert ele.OverloadedAlarm >= 0 and ele.OverloadedAlarm <= 2.0
            assert ele.DoorState >= 0 and ele.DoorState <= 1
            assert ele.DoorIsClosing in [True, False]
            assert ele.DoorIsOpening in [True, False]
            for t in ele.ReservedTargetFloors:
                assert t >= 1 and t <= ele.MaximumFloor
            
            #relation
            if(ele_Velocity == 0 and ele.Direction != 0):
                assert (ele_Floor % 1) == 0 or \
                        (ele_Floor % 1 != 0 and next_ele.Direction == 0)
            if(round(ele_Floor, 1) % 1 != 0 and ele.Direction != 0):
                assert ele_Velocity != 0 or next_ele_Velocity != 0 or\
                         next_ele.Direction == 0 or ele_Floor == ele.CurrentDispatchTarget
            assert (ele.DoorIsClosing and ele.DoorIsOpening) == False
            if(ele.DoorState < 1 and ele.DoorState > 0):
                assert (ele.DoorIsClosing or ele.DoorIsOpening) == True  
                assert ele_Floor % 1 == 0
            # if(ele.DoorState in [0.0, 1.0]):
            #     assert (ele.DoorIsClosing or ele.DoorIsOpening) == False  # ignore
            if(ele.DoorState in [0.0, 1.0]):
                if((ele.DoorIsClosing or ele.DoorIsOpening) == True):
                    if(next_ele.DoorState in [0.0, 1.0]):
                        assert (next_ele.DoorIsClosing or next_ele.DoorIsOpening) == False
            if((ele_Floor % 1 != 0) or ((ele.DoorIsClosing and ele.DoorIsOpening) == True)):
                assert ele.DoorState == 0.0
                assert ele.DoorIsClosing == False or next_ele.DoorIsClosing == False
                assert ele.DoorIsOpening == False
            if(ele_Velocity != 0.0 and ele.Direction != 0):
                assert ele.DoorState == 0.0
            if(ele_Velocity != 0.0 and len(ele.ReservedTargetFloors) > 0):
                assert ele_LoadWeight > 0
            if(ele_Velocity != 0.0 and ele_LoadWeight > 0):
                assert len(ele.ReservedTargetFloors) > 0
            if(next_ele.OverloadedAlarm > 0 and ele.OverloadedAlarm == 0):
                assert next_ele_LoadWeight >= ele.MaximumLoad - 200
            if(len(ele.ReservedTargetFloors) != 0):
                assert ele_LoadWeight >= 20

            # dynamic check
            delta_Floor = round(next_ele_Floor - ele_Floor, 2)
            assert delta_Floor * next_ele_Velocity >= 0 or delta_Floor * ele_Velocity >= 0
            target_list = ele.ReservedTargetFloors[:]
            # if(ele.CurrentDispatchTarget != 0):
            #     target_list.append(ele.CurrentDispatchTarget)
            if(delta_Floor > 0 and ele_Velocity != 0.0 and ele_Floor % 1 != 0): # going up
                min_target = min(target_list) if len(target_list) > 0 else ele.MaximumFloor + 1
                assert ele_Floor <= min_target
                assert next_ele_Velocity > 0 or ele_Velocity > 0 or ele.Direction == 0
            if(delta_Floor < 0 and ele_Velocity != 0.0 and ele_Floor % 1 != 0): # going down
                max_target = max(target_list) if len(target_list) > 0 else 0
                assert ele_Floor >= max_target
                assert next_ele_Velocity < 0 or ele_Velocity < 0 or ele.Direction == 0
            # if(delta_Floor == 0):
            #     assert next_ele_Velocity == 0 or ele_Velocity * next_ele_Velocity <= 0
            
            if((next_ele_LoadWeight - ele_LoadWeight) > 0.01):
                assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            if((next_ele_LoadWeight - ele_LoadWeight) < -0.01):
                assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            if(ele.OverloadedAlarm < next_ele.OverloadedAlarm):
                assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
                assert len(next_ele.ReservedTargetFloors) == len(ele.ReservedTargetFloors) #?????
                # assert next_ele_LoadWeight >= ele_LoadWeight # not right

            if(len(next_ele.ReservedTargetFloors) > len(ele.ReservedTargetFloors)):
                assert (next_ele_LoadWeight - ele_LoadWeight) >= 0 #!!!
                assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing
            if(len(next_ele.ReservedTargetFloors) < len(ele.ReservedTargetFloors)):
                # assert (next_ele_LoadWeight - ele_LoadWeight) < 0 # not right
                assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing

            # if(ele.OverloadedAlarm > 0):
            #     assert ele.ReservedTargetFloors == next_ele.ReservedTargetFloors
            #     assert ele_LoadWeight == next_ele_LoadWeight
            #     assert ele.DoorState > 0 or ele.DoorIsOpening or ele.DoorIsClosing



        if(fail_flag):
            stop_count -= 1
            if(stop_count == 0):
                print('\n\nSome error appear before several steps, please check\n\n')
                exit(1)

    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        print('An error occurred on line {} in statement {}'.format(line, text))
        print('\n========================== ele num: ', i)
        print('\nlast: ', ele)
        print('\nthis: ', next_ele)
        print('\n========================== please check\n\n')
        fail_flag = True
        # print('==========================next state')
        # print_next_state(next_state)
        # exit(1)

def print_state(state, action):
    assert isinstance(state, MansionState)
    print('Num\tact\tact.dir\tFloor\t\tMaxF\tV\t\tMaxV\tDir\tTarget\tTDir\tLoad\tMaxL\tOver\tDoor\topening\tclosing\tReservedTargetFloors')
    i = 0
    for i in range(len(state.ElevatorStates)):
        ele = state.ElevatorStates[i]
        act = action[i]
        assert isinstance(ele, ElevatorState)
        assert isinstance(act, ElevatorAction)
        print(i,"\t|",act.TargetFloor,"\t|",act.DirectionIndicator,"\t|",
                    '%2.4f'%ele.Floor,"\t|",ele.MaximumFloor,"\t|",
                    '%2.7f'%ele.Velocity,"\t|",ele.MaximumSpeed,"\t|",
                    ele.Direction,"\t|",ele.CurrentDispatchTarget,"\t|",ele.DispatchTargetDirection,"\t|",
                    int(ele.LoadWeight),"\t|",ele.MaximumLoad,"\t|",'%.2f'%ele.OverloadedAlarm,"\t|",
                    ele.DoorState,"\t|",int(ele.DoorIsOpening),"\t|",int(ele.DoorIsClosing),"\t|",ele.ReservedTargetFloors)
        i += 1
    print('------------------RequiringUpwardFloors', state.RequiringUpwardFloors)
    print('------------------RequiringDownwardFloors', state.RequiringDownwardFloors)
    print('')
    # time.sleep(2)

def print_next_state(state):
    assert isinstance(state, MansionState)
    print('Num\tact\tact.dir\tFloor\t\tMaxF\tV\tMaxV\tDir\tTarget\tTDir\tLoad\tMaxL\tOver\tDoor\topening\tclosing\tRT')
    i = 0
    for i in range(len(state.ElevatorStates)):
        ele = state.ElevatorStates[i]
        # act = action[i]
        assert isinstance(ele, ElevatorState)
        # assert isinstance(act, ElevatorAction)
        i += 1
        print(i,"\t|",' ',"\t|",' ',"\t|",
                    '%.2f'%ele.Floor,"\t|",ele.MaximumFloor,"\t|",
                    '%.1f'%ele.Velocity,"\t|",ele.MaximumSpeed,"\t|",
                    ele.Direction,"\t|",ele.CurrentDispatchTarget,"\t|",ele.DispatchTargetDirection,"\t|",
                    '%.1f'%ele.LoadWeight,"\t|",ele.MaximumLoad,"\t|",ele.OverloadedAlarm,"\t|",
                    ele.DoorState,"\t|",int(ele.DoorIsOpening),"\t|",int(ele.DoorIsClosing),"\t|",ele.ReservedTargetFloors)
    print('------------------RequiringUpwardFloors', state.RequiringUpwardFloors)
    print('------------------RequiringDownwardFloors', state.RequiringDownwardFloors)
    print('')
    # time.sleep(2)




def run_mansion_main(mansion_env, policy_handle, iteration):
    last_state = mansion_env.reset()
    # policy_handle.link_mansion(mansion_env.attribute)
    # policy_handle.load_settings()
    i = 0
    acc_reward = 0.0

     # = copy.deepcopy(mansion_env.state)

    while i < iteration:
        i += 1
        # state = mansion_env.state

        action = policy_handle.policy(last_state)
        state, r, _, _ = mansion_env.step(elevatoraction_to_list(action))
        # output_info = policy_handle.feedback(last_state, action, r)
        acc_reward += r

        # if(isinstance(output_info, dict) and len(output_info) > 0):
        #     mansion_env.log_notice("%s", output_info)
        if(i % 3600 == 0):
            print(
                "Accumulated Reward: %f, Mansion Status: %s",
                acc_reward, mansion_env.statistics)
            acc_reward = 0.0

        print_state(state, action)
        print('reward: %f' % r)
        state_check(last_state, state, action)
        last_state = copy.deepcopy(state)


# run main program with args
def run_qa_test(configfile, iterations, controlpolicy, set_seed=None):
    print('configfile:', configfile) # configuration file for running elevators
    print('iterations:', iterations) # total number of iterations
    print('controlpolicy:', controlpolicy) # policy type: rule_benchmark or others

    mansion_env = LiftSim(configfile)

    if(set_seed):
        mansion_env.seed(set_seed)

    if controlpolicy == 'rule_benchmark':
        dispatcher = Rule_dispatcher(mansion_env, iterations)
    elif controlpolicy == 'rl_benchmark':
        pass

    run_mansion_main(mansion_env, dispatcher, iterations)    

    return 0

def run_time_step_abnormal_test(configfile, iterations, controlpolicy, set_seed=None):
    try:
        run_qa_test(configfile, iterations, controlpolicy, set_seed=set_seed)
    except AssertionError:
        print('run_time_step_abnormal_test pass')

def run_action_abnormal_test(action_target_floor, action_target_direction, set_seed):
    flag = True
    try:
        env = LiftSim()
        if(set_seed):
            env.seed(set_seed)
        state = env.reset()

        action = [ElevatorAction(action_target_floor, action_target_direction) for i in range(4)]
        next_state, reward, _, _ = env.step(elevatoraction_to_list(action))
    except AssertionError:
        flag = False
        print('abnormal action: ', action_target_floor, type(action_target_floor) \
                                , action_target_direction, type(action_target_direction))
        print('run_action_abnormal_test pass')
    if (flag):
        print('abnormal action: ', action_target_floor, type(action_target_floor) \
                                , action_target_direction, type(action_target_direction))
        print('run_action_abnormal_test fail')
        assert False

def elevatoraction_to_list(action):
    action_list = []
    for a in action:
        action_list.append(a.TargetFloor)
        action_list.append(a.DirectionIndicator)
    return action_list

if __name__ == "__main__":
    if (len(sys.argv) == 2):
        set_seed = int(sys.argv[1])
    else:
        set_seed = None

    run_time_step_abnormal_test('rlschool/liftsim/tests/conf/config_time_step_more_than_1.ini', 100, 'rule_benchmark', set_seed)
    run_action_abnormal_test(-2, 1, set_seed)
    run_action_abnormal_test(10000, -1, set_seed)
    run_action_abnormal_test(5.0, 1, set_seed)
    run_action_abnormal_test('5', 1, set_seed)
    run_action_abnormal_test(5, 4, set_seed)
    run_action_abnormal_test(5, '-1', set_seed)
    run_qa_test('rlschool/liftsim/config.ini', 4000, 'rule_benchmark', set_seed)
    run_qa_test('rlschool/liftsim/tests/conf/config1.ini', 4000, 'rule_benchmark', set_seed) # 1 elevator
    run_qa_test('rlschool/liftsim/tests/conf/config2.ini', 4000, 'rule_benchmark', set_seed) # 100 floors 20 elevator 0.3 time_step
    run_qa_test('rlschool/liftsim/tests/conf/config3.ini', 4000, 'rule_benchmark', set_seed) # quick person generator
    run_qa_test('rlschool/liftsim/tests/conf/config4.ini', 4000, 'rule_benchmark', set_seed) # 1.0 time_step
