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
from collections import deque

from rlschool.liftsim.environment.mansion.elevator import Elevator
from rlschool.liftsim.environment.mansion.utils import PersonType
from rlschool.liftsim.environment.mansion.utils import MansionAttribute, MansionState, ElevatorState
from rlschool.liftsim.environment.mansion.utils import EPSILON, ENTERING_TIME
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from rlschool.liftsim.environment.mansion.person_generators.generator_proxy import PersonGenerator
from rlschool.liftsim.environment.mansion.person_generators.person_generator import PersonGeneratorBase


class MansionManager(object):
    """
    Mansion Class
    Mansion Randomly Generates Person that requiring elevators for a lift
    """

    def __init__(
            self,
            elevator_number,
            person_generator,
            mansion_config,
            name="Mansion"):
        """
        Initializing the Building
        Args:
          floor_number: number of floor in the building
          elevator_number: number of elevator in the building
          floor_height: height of single floor
          time_step:    Simulation timestep
          person_generator: PersonGenerator class that generates stochastic pattern of person flow
        Returns:
          None
        """
        assert isinstance(mansion_config, MansionConfig)
        assert isinstance(person_generator, PersonGeneratorBase)
        self._name = name
        self._config = mansion_config
        self._floor_number = self._config.number_of_floors
        self._floor_height = self._config.floor_height
        self._dt = self._config.delta_t
        self._elevator_number = elevator_number
        self._person_generator = person_generator

        # if people are waiting for more than 300 seconds, he would give up!
        self._given_up_time_limit = 300

        #used for statistics
        self._statistic_interval = int(600 / self._dt)
        self._delivered_person = deque()
        self._generated_person = deque()
        self._abandoned_person = deque()
        self._cumulative_waiting_time = deque()
        self._cumulative_energy_consumption = deque()

        self.reset_env()

    def reset_env(self):
        self._elevators = []
        for i in range(self._elevator_number):
            self._elevators.append(
                Elevator(start_position=0.0,
                         mansion_config=self._config,
                         name="%s_E%d" % (self._name, i + 1)))

        self._config.reset()
        self._person_generator.link_mansion(self._config)

        # whether the go up/down button is clicked
        self._button = [[False, False] for i in range(self._floor_number)]
        self._wait_upward_persons_queue = [
            deque() for i in range(
                self._floor_number)]
        self._wait_downward_persons_queue = [
            deque() for i in range(
                self._floor_number)]


    @property
    def state(self):
        """
        Return Current state of the building simulator
        """
        upward_req = []
        downward_req = []
        state_queue = []
        for idx in range(self._floor_number):
            if(self._button[idx][0]):
                upward_req.append(idx + 1)
            if(self._button[idx][1]):
                downward_req.append(idx + 1)
        for i in range(self._elevator_number):
            state_queue.append(self._elevators[i].state)

        return MansionState(state_queue, upward_req, downward_req)

    def run_mansion(self, actions):
        """
        Perform one step of simulations
        Args:
          actions: A list of actions, e.g., action.add_target = [2, 6, 8], action.remove_target = [4]
          mark the target floor to be added into the queue or removed from the queue
        Returns:
          State, Cumulative Wating Time for Person, Energy Consumption of Elevator
        """
        self._config.step()  # update the current time

        person_list = self._person_generator.generate_person()
        tmp_generated_person = len(person_list)
        for person in person_list:
            if(person.SourceFloor < person.TargetFloor):
                self._wait_upward_persons_queue[person.SourceFloor -
                                                1].appendleft(person)
            elif(person.SourceFloor > person.TargetFloor):
                self._wait_downward_persons_queue[person.SourceFloor - 1].appendleft(
                    person)
        energy_consumption = [0.0 for i in range(self._elevator_number)]

        # carry out actions on each elevator
        for idx in range(self._elevator_number):
            self._elevators[idx].set_action(actions[idx])

        # make each elevator run one step
        loaded_person_num = 0
        tmp_delivered_person = 0
        for idx in range(self._elevator_number):
            energy_consumption[idx], delivered_person_time, tmp_loaded_person = self._elevators[idx].run_elevator()
            tmp_delivered_person += len(delivered_person_time)
            loaded_person_num += tmp_loaded_person

        for floor in range(self._floor_number):
            if(len(self._wait_upward_persons_queue[floor]) > 0):
                self._button[floor][0] = True
            else:
                self._button[floor][0] = False

            if(len(self._wait_downward_persons_queue[floor]) > 0):
                self._button[floor][1] = True
            else:
                self._button[floor][1] = False

        ele_idxes = [i for i in range(self._elevator_number)]
        random.shuffle(ele_idxes)

        for ele_idx in ele_idxes:
            floor, delta_distance = self._elevators[ele_idx].nearest_floor
            is_open = self._elevators[ele_idx].is_fully_open and (
                abs(delta_distance) < 0.05)
            is_ready = self._elevators[ele_idx].ready_to_enter
            # Elevator stops at certain floor and the direction is consistent
            # with the customers' target direction
            floor_idx = floor - 1

            if(is_open):
                if(self._elevators[ele_idx]._direction == 1):
                    self._button[floor_idx][0] = False
                elif(self._elevators[ele_idx]._direction == -1):
                    self._button[floor_idx][1] = False

            if(is_ready and is_open):
                self._config.log_debug(
                    "Floor: %d, Elevator: %s is open, %d persons are waiting to go upward, %d downward", floor, self._elevators[ele_idx].name, len(
                        self._wait_upward_persons_queue[floor_idx]), len(
                        self._wait_downward_persons_queue[floor_idx]))

                if(self._elevators[ele_idx]._direction == -1):
                    for i in range(
                            len(self._wait_downward_persons_queue[floor_idx]) - 1, -1, -1):
                        entering_person = self._wait_downward_persons_queue[floor_idx][i]
                        req_succ = self._elevators[ele_idx].person_request_in(
                            entering_person)
                        if(req_succ):
                            del self._wait_downward_persons_queue[floor_idx][i]
                            self._config.log_debug(
                                "Person %s is walking into the %s elevator",
                                entering_person,
                                self._elevators[ele_idx].name)
                        else:  # if the reason of fail is overweighted, try next one
                            if not self._elevators[ele_idx]._is_overloaded_alarm:
                                break
                elif(self._elevators[ele_idx]._direction == 1):
                        # if no one is entering
                    for i in range(
                            len(self._wait_upward_persons_queue[floor_idx]) - 1, -1, -1):
                        entering_person = self._wait_upward_persons_queue[floor_idx][i]
                        req_succ = self._elevators[ele_idx].person_request_in(
                            entering_person)
                        if(req_succ):
                            del self._wait_upward_persons_queue[floor_idx][i]
                            self._config.log_debug(
                                "Person %s is walking into the %s elevator",
                                entering_person,
                                self._elevators[ele_idx].name)
                        else:
                            if not self._elevators[ele_idx]._is_overloaded_alarm:
                                break

        # Remove those who waited too long
        give_up_persons = 0
        for floor_idx in range(self._floor_number):
            for pop_idx in range(
                    len(self._wait_upward_persons_queue[floor_idx]) - 1, -1, -1):
                if(self._config.raw_time - self._wait_upward_persons_queue[floor_idx][pop_idx].AppearTime > self._given_up_time_limit):
                    self._wait_upward_persons_queue[floor_idx].pop()
                    give_up_persons += 1
                else:
                    break
            for pop_idx in range(
                    len(self._wait_downward_persons_queue[floor_idx]) - 1, -1, -1):
                if(self._config.raw_time - self._wait_downward_persons_queue[floor_idx][pop_idx].AppearTime > self._given_up_time_limit):
                    self._wait_downward_persons_queue[floor_idx].pop()
                    give_up_persons += 1
                else:
                    break

        cumulative_waiting_time = 0
        for i in range(self._floor_number):
            cumulative_waiting_time += self._dt * \
                len(self._wait_upward_persons_queue[i])
            cumulative_waiting_time += self._dt * \
                len(self._wait_downward_persons_queue[i])
        cumulative_waiting_time += loaded_person_num * self._dt
        cumulative_energy_consumption = float(sum(energy_consumption))

        self._delivered_person.appendleft(tmp_delivered_person)
        self._generated_person.appendleft(tmp_generated_person)
        self._abandoned_person.appendleft(give_up_persons)
        self._cumulative_waiting_time.appendleft(cumulative_waiting_time)
        self._cumulative_energy_consumption.appendleft(cumulative_energy_consumption)
        if(len(self._delivered_person) > self._statistic_interval):
          self._delivered_person.pop()
          self._generated_person.pop()
          self._abandoned_person.pop()
          self._cumulative_waiting_time.pop()
          self._cumulative_energy_consumption.pop()

        return cumulative_waiting_time, cumulative_energy_consumption, give_up_persons

    def get_statistics(self):
        """
        Get Mansion Statistics
        """
        return {
            "DeliveredPersons(10Minutes)": int(sum(self._delivered_person)),
            "GeneratedPersons(10Minutes)": int(sum(self._generated_person)),
            "AbandonedPersons(10Minutes)": int(sum(self._abandoned_person)),
            "EnergyConsumption(10Minutes)": float(sum(self._cumulative_energy_consumption)),
            "TotalWaitingTime(10Minutes)": float(sum(self._cumulative_waiting_time))}

    @property
    def attribute(self):
        """
        returns all kinds of attributes
        """
        return MansionAttribute(
            self._elevator_number,
            self._floor_number,
            self._floor_height)

    @property
    def config(self):
        """
        Returns config of the mansion
        """
        return self._config

    @property
    def waiting_queue(self):
        """
        Returns the waiting queue of each floor
        """
        return [self._wait_upward_persons_queue, self._wait_downward_persons_queue]

    @property
    def loaded_people(self):
        """
        Returns: the number of loaded people of each elevator
        """
        return [self._elevators[i].loaded_people_num for i in range(self._elevator_number)]

    @property
    def name(self):
        """
        Returns name of the mansion
        """
        return self._name
