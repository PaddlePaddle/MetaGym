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

import sys
import random
import queue
from rlschool.liftsim.environment.mansion.utils import ElevatorState, ElevatorAction, MansionState
from rlschool.liftsim.environment.mansion.utils import EPSILON, HUGE


class Rule_dispatcher():
    """
    A rule benchmark demonstration of the dispatcher
    A dispatcher must provide policy and feedback function
    The policy function receives MansionState and output ElevatorAction Lists
    The feedback function receives reward
    """

    def __init__(self, env, max_episode):
        self.elevator_num = env.attribute.ElevatorNumber
        self.max_episode = max_episode
        self.env = env

    def run_dispacher(self):
        self.elevator_num = self.env.attribute.ElevatorNumber
        running_step_counter = 0
        acc_reward = 0.0
        
        while running_step_counter < self.max_episode:
            self.env.render()
            running_step_counter += 1
            state = self.env.state
            action = self.policy(state)
            _, reward, done, _ = self.env.step(action)
            acc_reward += reward

            if(running_step_counter % 3600 == 0):
                self.env.log_notice(
                    "Accumulated Reward: %f, Mansion Status: %s",
                    acc_reward, self.env.statistics)
                acc_reward = 0.0

    def policy(self, state):
        ret_actions = [
            ElevatorAction(
                0, 1) for i in range(self.elevator_num)]

        idle_ele_queue = queue.Queue()
        upward_floor_address_dict = dict()
        downward_floor_address_dict = dict()

        for i in range(len(state.ElevatorStates)):
            idle_ele_queue.put(i)

        for floor in state.RequiringUpwardFloors:
            # Addressing Elevator, Priority
            upward_floor_address_dict[floor] = (-1, -HUGE)

        for floor in state.RequiringDownwardFloors:
            downward_floor_address_dict[floor] = (-1, -HUGE)

        while not idle_ele_queue.empty():
            sel_ele = idle_ele_queue.get()
            if(state.ElevatorStates[sel_ele].Direction > 0):
                assigned = False
                sel_priority = -HUGE
                sel_floor = -1
                for upward_floor in state.RequiringUpwardFloors:
                    if(upward_floor < state.ElevatorStates[sel_ele].Floor - EPSILON):
                        continue
                    priority = state.ElevatorStates[sel_ele].Floor - \
                        upward_floor
                    if(upward_floor in state.ElevatorStates[sel_ele].ReservedTargetFloors):
                        priority = min(0.0, priority + 5.0)
                    if (state.ElevatorStates[sel_ele].Velocity < EPSILON):
                        priority -= 5.0
                    if(priority > upward_floor_address_dict[upward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_floor = upward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(sel_floor, 1)
                    if(upward_floor_address_dict[sel_floor][0] >= 0):
                        ret_actions[upward_floor_address_dict[sel_floor]
                                    [0]] = ElevatorAction(0, 1)
                        idle_ele_queue.put(
                            upward_floor_address_dict[sel_floor][0])
                    upward_floor_address_dict[sel_floor] = (
                        sel_ele, sel_priority)
                    assigned = True

                # In case no floor is assigned to the current elevator, we
                # search all requiring downward floor, find the largest floor
                # and assign it to the elevator
                if(not assigned):
                    if(len(state.RequiringDownwardFloors) > 0):
                        max_unassigned_down_floor = -1
                        for downward_floor in state.RequiringDownwardFloors:
                            if(downward_floor_address_dict[downward_floor][0] < 0 and max_unassigned_down_floor < downward_floor):
                                max_unassigned_down_floor = downward_floor
                        if(max_unassigned_down_floor >= 0):
                            ret_actions[sel_ele] = ElevatorAction(
                                max_unassigned_down_floor, -1)
                            priority = - \
                                state.ElevatorStates[sel_ele].Floor - EPSILON + max_unassigned_down_floor
                            downward_floor_address_dict[max_unassigned_down_floor] = (
                                sel_ele, priority)

            if(state.ElevatorStates[sel_ele].Direction < 0):
                assigned = False
                sel_priority = -HUGE
                sel_floor = -1
                for downward_floor in state.RequiringDownwardFloors:
                    if(downward_floor > state.ElevatorStates[sel_ele].Floor + EPSILON):
                        continue
                    priority = - \
                        state.ElevatorStates[sel_ele].Floor + downward_floor
                    if(downward_floor in state.ElevatorStates[sel_ele].ReservedTargetFloors):
                        priority = min(0.0, priority + 5.0)
                    if (state.ElevatorStates[sel_ele].Velocity < EPSILON):
                        priority -= 5.0
                    if(priority > downward_floor_address_dict[downward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_floor = downward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(sel_floor, 1)
                    if(downward_floor_address_dict[sel_floor][0] >= 0):
                        ret_actions[downward_floor_address_dict[sel_floor][0]] = ElevatorAction(
                            0, 1)
                        idle_ele_queue.put(
                            downward_floor_address_dict[sel_floor][0])
                    downward_floor_address_dict[sel_floor] = (
                        sel_ele, sel_priority)
                    assigned = True

                # In case no floor is assigned to the current elevator, we
                # search all requiring-upward floor, find the lowest floor and
                # assign it to the elevator
                if(not assigned):
                    if(len(state.RequiringUpwardFloors) > 0):
                        min_unassigned_up_floor = HUGE
                        for upward_floor in state.RequiringUpwardFloors:
                            if(upward_floor_address_dict[upward_floor][0] < 0 and min_unassigned_up_floor > upward_floor):
                                min_unassigned_up_floor = upward_floor
                        if(min_unassigned_up_floor >= 0 and min_unassigned_up_floor < HUGE - 1):
                            ret_actions[sel_ele] = ElevatorAction(
                                min_unassigned_up_floor, 1)
                            priority = state.ElevatorStates[sel_ele].Floor + \
                                EPSILON - min_unassigned_up_floor
                            upward_floor_address_dict[min_unassigned_up_floor] = (
                                sel_ele, priority)

            if(state.ElevatorStates[sel_ele].Direction == 0):
                # in case direction == 0,  select the closest requirements
                sel_floor = -1
                sel_priority = -HUGE
                sel_direction = 0

                for upward_floor in state.RequiringUpwardFloors:
                    priority = -abs(upward_floor -
                                    state.ElevatorStates[sel_ele].Floor)
                    if(priority > upward_floor_address_dict[upward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_direction = 1
                        sel_floor = upward_floor

                for downward_floor in state.RequiringDownwardFloors:
                    priority = -abs(downward_floor -
                                    state.ElevatorStates[sel_ele].Floor)
                    if(priority > downward_floor_address_dict[downward_floor][1] and priority > sel_priority):
                        sel_priority = priority
                        sel_direction = - 1
                        sel_floor = downward_floor

                if(sel_floor > 0):
                    ret_actions[sel_ele] = ElevatorAction(
                        sel_floor, sel_direction)
                    if(sel_direction > 0):
                        if(upward_floor_address_dict[sel_floor][0] >= 0):
                            idle_ele_queue.put(
                                upward_floor_address_dict[sel_floor][0])
                        upward_floor_address_dict[sel_floor] = (
                            sel_ele, sel_priority)
                    else:
                        if(downward_floor_address_dict[sel_floor][0] >= 0):
                            idle_ele_queue.put(
                                downward_floor_address_dict[sel_floor][0])
                        downward_floor_address_dict[sel_floor] = (
                            sel_ele, sel_priority)

        # print min_unaddressed_up_lift, max_unaddressed_down_lift,
        # state.RequiringUpwardFloors, state.RequiringDownwardFloors
        # action_list = []
        # for a in ret_actions:
        #     action_list.append(a.TargetFloor)
        #     action_list.append(a.DirectionIndicator)
        return ret_actions
