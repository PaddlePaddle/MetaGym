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
from rlschool.liftsim.environment.mansion.utils import PersonType, ElevatorState, ElevatorAction
from rlschool.liftsim.environment.mansion.utils import EPSILON, GRAVITY
from rlschool.liftsim.environment.mansion.utils import velocity_planner
from rlschool.liftsim.environment.mansion.mansion_config import MansionConfig
from copy import deepcopy


class Elevator(object):
    """
    A simulator of elevator motion and power consumption
    Energy consumption calculated according to the paper

    Adak, M. Fatih, Nevcihan Duru, and H. Tarik Duru.
    "Elevator simulator design and estimating energy consumption of an elevator system."
    Energy and Buildings 65 (2013): 272-280.
    """

    def __init__(self,
                 start_position,
                 mansion_config,
                 name="ELEVATOR"):
        assert isinstance(mansion_config, MansionConfig)
        self._number_of_floors = mansion_config.number_of_floors
        self._floor_height = mansion_config.floor_height
        self._current_position = start_position
        self._maximum_speed = mansion_config.maximum_speed
        self._maximum_acceleration = mansion_config.maximum_acceleration
        self._maximum_capacity = mansion_config.maximum_capacity
        self._door_open_close_velocity = 1.0 / mansion_config.door_opening_closing_time
        self._person_entering_time = mansion_config._person_entering_time
        self._mpee_number = mansion_config._mpee_number
        self._dt = mansion_config.delta_t

        self._load_weight = 0.0
        self._current_velocity = 0.0
        self._door_open_rate = 0.0
        self._current_time = 0.0
        self._is_door_opening = False
        self._keep_door_open_left = 0.0
        self._keep_door_open_lag = 2.0 + self._dt
        # actually the lag is 2.0s, as the value will decrease dt in the same
        # time step
        self._is_door_closing = False
        self._config = mansion_config

        # A list of target floors in descending / ascending order
        self._target_floors = list()
        # User clicked buttons, if clicked, never deactivate before arriving at
        # that floor!
        self._clicked_buttons = set()
        # Target of dispatcher, 0 if no target needs to be specified
        self._dispatch_target = 0
        # Specify the direction of the elevator when arrived at the target
        # It only works when direction of the elevator _direction = 0
        self._dispatch_target_direction = 1
        # Loaded Persons Queue
        self._loaded_person = [list() for i in range(self._number_of_floors)]
        # Entering Person
        self._entering_person = list()
        self._exiting_person = list()

        self._is_unloading = False
        self._is_unentering = False

        # Predefined direction of the elvator:  0 - stop; 1 - Go Up; 2 - Go
        # Down
        self._direction = 0

        self._name = name
        self._is_overloaded_alarm = 0.0

        self._reaction_distance = 0.5 * \
            (self._maximum_speed * self._maximum_speed) / \
            self._maximum_acceleration

    def __repr__(self):
        return ("""Elevator Object: %s\n
            State\n\t
            Position: %f\n\t
            Floors: %2.2f\n\t
            Velocity: %f\n\t
            Load: %f\n\t
            Current Direction: %d\n\t
            Dispatch Target: %d\n\t
            Dispatch Target Direction: %d\n\t
            Reserved Target Floors: %s\n\t
            Button: %s\n\t
            Is Overloaded: %d\n\t
            Door Open Rate: %f\n\t
            Is Door Opening: %d\n\t
            Is Door Closing: %d\n\t
            Loaded Persons: %s\n\t
            Entering Persons: %s\n\t
            Exiting Persons: %s\n\t"""
            ) % (self._name,
                    self._current_position, 
                    self._current_position / self._floor_height + 1.0,
                    self._current_velocity, 
                    self._load_weight, 
                    self._direction,
                    self._dispatch_target, 
                    self._dispatch_target_direction, 
                    self._target_floors, 
                    self._clicked_buttons,
                    self._is_overloaded_alarm, 
                    self._door_open_rate, 
                    self._is_door_opening, 
                    self._is_door_closing, 
                    self._loaded_person, 
                    self._entering_person, 
                    self._exiting_person
                )

    @property
    def door_fully_open(self, floor):
        """
        Returns whether the door is fully open in the corresponding floor
        Args:
          floor, the floor to be queried
        Returns:
          True if the elevator stops at the floor and opens the door, False in other case
        """
        cur_floor = self._current_position / self._floor_height + 1
        if(abs(cur_floor - float(floor)) < EPSILON
                and self._door_open_rate > 1.0 - EPSILON):
            return True
        else:
            return False

    @property
    def name(self):
        """
        Return Name of the Elevator
        """
        return self._name

    @property
    def state(self):
        """
        Return Formalized States
        """
        floor = self._current_position / self._floor_height + 1
        return ElevatorState(
            floor,
            self._number_of_floors,
            self._current_velocity,
            self._maximum_speed,
            self._direction,
            self._door_open_rate,
            self._dispatch_target,
            self._dispatch_target_direction,
            self._load_weight,
            self._maximum_capacity,
            self._target_floors,
            self._is_overloaded_alarm,
            self._is_door_opening,
            self._is_door_closing)

    @property
    def nearest_floor(self):
        """
        Get the nearest_floor
        Args:
          None
        Returns:
          Nearest Floor
        """
        cur_floor = self._current_position / self._floor_height + 1.0
        nearest_floor = int(cur_floor + 0.5)
        delta_distance = nearest_floor - float(cur_floor)
        return nearest_floor, delta_distance

    @property
    def nearest_reachable_floor(self):
        """
        Get the nearest reachable floor
        Args:
          None
        Returns:
          Nearest reachable floor
        """
        if(self._current_velocity < 0.0):
            velocity_sign = -1
        else:
            velocity_sign = 1

        reaction_distance = 0.5 * self._current_velocity * \
            self._current_velocity / self._maximum_acceleration
        minimum_stop_position = self._current_position + velocity_sign * reaction_distance
        # print("finding the minimum stop position,", minimum_stop_position)
        return int(minimum_stop_position / self._floor_height + 2)

    @property
    def is_fully_open(self):
        return self._door_open_rate > 1.0 - EPSILON

    @property
    def ready_to_enter(self):
        """
        Returns:
          whether it is OK to enter the elevator
        """
        return (not self._is_unloading) and (not self._is_entering)

    @property
    def is_stopped(self):
        """
        Returns:
          whether the elevator stops
        """
        return (abs(self._current_velocity) < EPSILON)

    @property
    def loaded_people_num(self):
        """
        Returns:
            the number of loaded_people
        """
        return sum(len(self._loaded_person[i]) for i in range(self._number_of_floors))

    def _check_floor(self, floor):
        """
        check if the floor is in valid range
        """
        if(floor < 1 or floor > self._number_of_floors):
            return False
        return True

    def _clip_v(self, v):
        """
        Clip the velocity
        """
        return max(-self._maximum_speed, min(self._maximum_speed, v))

    def _check_abnormal_state(self):
        """
        the function checks that the elevator is OK
        raise Exceptions if anything is wrong
        """
        cur_floor = self._current_position / self._floor_height + 1.0
        if(cur_floor < 1 - EPSILON or cur_floor > self._number_of_floors + EPSILON):
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, current_floor = %f",
                self._name,
                cur_floor)
        if(abs(self._current_velocity) > self._maximum_speed):
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, current_velocity = %f exceeds the maximum velocity %f",
                self._name,
                self._current_velocity,
                self._maximum_speed)
        if(self._load_weight > self._maximum_capacity):
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, load_weight(%f) > maximum capacity (%f)",
                self._name,
                self._load_weight,
                self._maximum_capacity)
        # check whether target floor is in proper arrangement
        if(len(self._target_floors) > 0):
            for i in range(len(self._target_floors) - 1):
                if(self._direction * (self._target_floors[i + 1] - self._target_floors[i]) < - EPSILON):
                    self._config.log_fatal(
                        "Abnormal State detected for elevator %s, direction(%d) do not match target floors %s",
                        self._name,
                        self._direction,
                        self._target_floors)
            if(self._direction * (self._target_floors[0] - cur_floor) < - EPSILON):
                self._config.log_fatal(
                    "Abnormal State detected for elevator %s, direction(%d) do not match target floors %s and current_floor %2.2f, %s",
                    self._name,
                    self._direction,
                    self._target_floors,
                    cur_floor,
                    self)
        if(self._direction * self._current_velocity < - 1.0):
            self._config.log_fatal(
                "Abnormal State detected for elevator %s, direction(%d) dot not match current velocity %f, Elevator state: %s",
                self._name,
                self._direction,
                self._current_velocity,
                self)

    def _check_target_validity(self, dispatch_target):
        """
        check whether a dispatch target is legal
        Args:
          target_floor: a new target to be pushed into the stop list to set as a target into list
        Returns:
          True - success False - Rejected
        """
        # 0: No Dispatch Target
        # if dispatch_target == 0:
        #  return True
        if(not self._check_floor(dispatch_target)):
            return False
        cur_floor = self._current_position / self._floor_height + 1.0
        if(abs(cur_floor - dispatch_target) < EPSILON and abs(self._current_velocity) < EPSILON):
            return True

        if(self._current_velocity < 0.0):
            velocity_sign = -1
        else:
            velocity_sign = 1

        # Need to consider the lag buffer
        # exp_v = self._current_velocity + velocity_sign * self._maximum_acceleration * self._dt
        # exp_v = self._clip_v(exp_v)
        # reaction_distance = (0.5 * exp_v * exp_v /
        # self._maximum_acceleration + self._dt * 0.5 * abs(exp_v +
        # self._current_velocity))
        reaction_distance = 0.5 * self._current_velocity * \
            self._current_velocity / self._maximum_acceleration

        target_position = (dispatch_target - 1) * self._floor_height
        minimum_stop_position = self._current_position + velocity_sign * reaction_distance

        if(self._direction > 0):
            # Target floor in oppsite direction, reject
            # if(target_position + 0.5 * EPSILON < self._current_position):
            #   return False
            # Check whether it is prompt to stop at the target floor
            if(target_position + 0.5 * EPSILON < minimum_stop_position):
                # If not, reject inserting
                return False
            return True
        elif(self._direction < 0):
            # Doing similar things to the opposite direction
            # if(target_position > self._current_position + 0.5 * EPSILON):
            #   return False
            if(target_position > minimum_stop_position + 0.5 * EPSILON):
                return False
            return True
        else:
            # in case direction = 0, it is always OK to insert
            return True

    def _insert_target(self, new_target):
        """
        insert a new target
        BETTER check the new_target with _check_target_validity() first before inserting!
        """
        if(self._direction >= 0):
            # If target floors are empty, insert directly
            if(len(self._target_floors) < 1):
                self._target_floors.insert(0, new_target)
            elif(new_target in self._target_floors):
                return
            else:
                # Else, find a position to insert, such that target floors are
                # in ascending order
                sel_i = len(self._target_floors)
                for i in range(len(self._target_floors)):
                    if(self._target_floors[i] == new_target):
                        return
                    elif(self._target_floors[i] > new_target):
                        sel_i = i
                        break
                if(sel_i >= 0):
                    self._target_floors.insert(sel_i, new_target)
        elif(self._direction < 0):
            # If target floors are empty, insert directly
            if(len(self._target_floors) < 1):
                self._target_floors.insert(0, new_target)
            elif(new_target in self._target_floors):
                return
            else:
                # Else, find a position to insert, such that target floors are
                # in descending order
                sel_i = len(self._target_floors)
                for i in range(len(self._target_floors)):
                    if(self._target_floors[i] == new_target):
                        return
                    elif(self._target_floors[i] < new_target):
                        sel_i = i
                        break
                if(sel_i >= 0):
                    self._target_floors.insert(sel_i, new_target)

    def _get_true_target(self):
        # print ("check target: %d, %d"%(self._dispatch_target, self._check_target_validity(self._dispatch_target)))
        if(self._is_overloaded_alarm > EPSILON):
            # in case overload alarm is ringing, neglect the dispatch target
            if(len(self._target_floors) < 1):
                return 0
            else:
                return self._target_floors[0]
        else:
            # Else, choose between dispatch target and target floor
            # check whether dispatch target is valid
            if(self._check_target_validity(self._dispatch_target)):
                if(len(self._target_floors) == 0):
                    # if self._dispatch_target == 0 and not self.is_stopped:
                    #  return self.nearest_reachable_floor
                    return self._dispatch_target
                elif(self._direction >= 0):
                    if(self._dispatch_target < self._target_floors[0]):
                        return self._dispatch_target
                    else:
                        return self._target_floors[0]
                elif(self._direction < 0):
                    if(self._dispatch_target > self._target_floors[0]):
                        return self._dispatch_target
                    else:
                        return self._target_floors[0]
            else:
                if(len(self._target_floors) == 0):
                    return 0
                else:
                    return self._target_floors[0]

    def run_elevator(self):
        """
        run elevator for one step
        Args:
          None
        Returns:
          Energy Consumption in one single step
        """

        # Get the current immediate target floor
        target_floor = self._get_true_target()
        if(target_floor <= 0):
            target_offset = 0.0
        else:
            target_offset = (target_floor - 1) * \
                self._floor_height - self._current_position
        no_reserved_target = (len(self._target_floors) < 1)
        # print ("target floor: %d" % target_floor)

        reaction_distance = 0.5 * self._current_velocity * \
            self._current_velocity / self._maximum_acceleration

        # Elevator stops at a target floor, stop and open_doors
        if(self.is_stopped and target_floor > 0):
            if(abs(target_offset) < EPSILON):
                if(self._door_open_rate > 1.0 - EPSILON):
                    if(target_floor in self._clicked_buttons):
                        self._clicked_buttons.remove(target_floor)
                    if(target_floor in self._target_floors):
                        self._target_floors.remove(target_floor)
                    self._dispatch_target = 0
                self.require_door_opening()

        floor_digit = self._current_position / self._floor_height + 1.0

        # if no target floors and current speeds = 0 and the door is closing,
        # set direction = 0
        if(no_reserved_target and self.is_stopped
                and self._door_open_rate < EPSILON):
            self._direction = 0

        # if the elevator is free, set the direction as specified by dispatcher
        if(self._direction == 0 and self.is_stopped):
            if(self._dispatch_target > 0 and
                    abs(self._dispatch_target - floor_digit) < EPSILON):
                if(self._dispatch_target_direction == 1 or self._dispatch_target_direction == - 1):
                    self._direction = self._dispatch_target_direction

        # Reaching the bottom or top of the floor, change the direction
        # instantly
        if(self.is_stopped and self._current_position <= EPSILON):
            self._direction = 1
        if(self.is_stopped and (self._current_position >= (self._number_of_floors - 1) * self._floor_height - EPSILON)):
            self._direction = -1

        if(len(self._target_floors) > 0 and self._direction == 0):
            if(self._target_floors[0] > floor_digit):
                self._direction = -1
            elif(self._target_floors[0] < floor_digit):
                self._direction = 1

        # check if there is any clicked button that misses in the target_floors
        # if there exists, add it to target floor
        for button in self._clicked_buttons:
            if(button not in self._target_floors):
                if(self._check_target_validity(button)):
                    self._insert_target(button)

        # check whether the target floor is achievable
        if(target_floor > 0):
            # recalculate target offset
            delta_floor = float(target_floor) - 1.0 - \
                self._current_position / self._floor_height
            if((delta_floor * self._direction) < - EPSILON):
                self._config.log_warning(
                    "Elevator: %s, Specified target floor is not achievable" +
                    ", delta_floor = %f, neglected floor = %d, information: %s",
                    self._name,
                    delta_floor,
                    target_floor,
                    self)
                if(target_floor in self._target_floors):
                    self._target_floors.remove(target_floor)

        # release the control command to the elevator door
        if(abs(self._current_velocity) > EPSILON):
            if(self._is_door_opening):
                self._is_door_opening = False
            elif(self._door_open_rate > EPSILON):
                self._is_door_closing = True

        # manage the control command of the door
        if(self._door_open_rate < EPSILON):
            self._is_door_closing = False
        elif(self._door_open_rate > 1.0 - EPSILON):
            self._is_door_opening = False
        if(len(self._entering_person) > 0 or len(self._exiting_person) > 0):
            self._is_door_closing = False
            if(self._door_open_rate < 1.0 - EPSILON):
                self._is_door_opening = True
                self._keep_door_open_left = self._keep_door_open_lag

        # manage the state of the door
        if(self._is_door_opening):
            self._door_open_rate = min(
                1.0, self._door_open_rate + self._door_open_close_velocity)
        elif(self._is_door_closing):
            self._door_open_rate = max(
                0.0, self._door_open_rate - self._door_open_close_velocity)

        # in case the door is open, print some log
        if(self._door_open_rate < EPSILON and self._is_door_closing):
            self._config.log_debug(
                "Elevator: %s, Door is fully closed, Elevator at %2.2f floor",
                self._name,
                self._current_position / self._floor_height + 1.0)
        if(self._door_open_rate > 1.0 - EPSILON and self._is_door_opening):
            self._config.log_debug(
                "Elevator: %s, Door is fully opening, Elevator at %2.2f floor",
                self._name,
                self._current_position / self._floor_height + 1.0)

        # if the elevator has the door open, hold the elevator still
        if(self._is_door_opening or self._is_door_closing or self._door_open_rate > EPSILON):
            hold_elevator = True
        else:
            hold_elevator = False

        # caculate the variables for energy loss
        acceleration = 0.0

        # The Elevator Dynamics
        eff_dt = self._dt
        if(hold_elevator):
            if(abs(self._current_velocity) < self._dt * self._maximum_acceleration):
                tmp_velocity = 0.0
                eff_dt = abs(self._current_velocity) / \
                    self._maximum_acceleration
            else:
                if(self._current_velocity > 0):
                    tmp_velocity = self._current_velocity - self._dt * self._maximum_acceleration
                else:
                    tmp_velocity = self._current_velocity + self._dt * self._maximum_acceleration
        else:
            tmp_velocity, eff_dt = velocity_planner(
                self._current_velocity, target_offset, self._maximum_acceleration, self._maximum_speed, self._dt)

        #print ("temp velocity", tmp_velocity, "eff dt", eff_dt, "target_offset:", target_offset, "current velocity", self._current_velocity)
        # update the elevator position
        self._current_position += 0.5 * \
            (tmp_velocity + self._current_velocity) * \
            eff_dt + tmp_velocity * (self._dt - eff_dt)
        # calculate the true acceleration
        acceleration = (tmp_velocity - self._current_velocity) / \
            max(eff_dt, EPSILON)
        
        force_1 = (self._config.net_weight + self._load_weight) * \
            (GRAVITY + acceleration)
        force_2 = self._config.rated_load * (GRAVITY - acceleration)
        net_force = force_1 - force_2
        m_load = abs(net_force) * self._config.pulley_radius / \
            self._config.motor_gear_ratio / self._config.gear_efficiency
        energy_consumption = (m_load *
                              abs((self._current_velocity+tmp_velocity)/2) /
                              self._config.gear_radius *
                              self._config.gear_efficiency /
                              self._config.motor_efficiency *
                              eff_dt +
                              self._config.standby_power_consumption *
                              self._dt)
 
        # update the elevator velocity
        self._current_velocity = tmp_velocity

        if(self._is_door_opening or self._is_door_closing):
            energy_consumption += self._config.automatic_door_power * self._dt

        self._is_overloaded_alarm = max(
            0.0, self._is_overloaded_alarm - self._dt)
        if(self.is_fully_open):
            self._keep_door_open_left = max(
                0.0, self._keep_door_open_left - self._dt)

        person_statistics = []
        # Unloading Persons
        remove_tmp_idx = list()
        for i in range(len(self._exiting_person)):
            self._exiting_person[i][1] -= self._dt
            if(self._exiting_person[i][1] < EPSILON):
                remove_tmp_idx.append(i)
        for i in sorted(remove_tmp_idx, reverse=True):
            out_person = self._exiting_person.pop(i)[0]
            self._load_weight -= out_person.Weight
            person_statistics.append(
                self._config.raw_time -
                out_person.AppearTime)

        # Loading Persons
        remove_tmp_idx = list()
        floor, delta_distance = self.nearest_floor
        floor_idx = floor - 1
        for i in range(len(self._entering_person)):
            self._entering_person[i][1] -= self._dt
            if(self._entering_person[i][1] < EPSILON):
                remove_tmp_idx.append(i)
        for i in sorted(remove_tmp_idx, reverse=True):
            entering_person = self._entering_person.pop(i)[0]
            self._loaded_person[entering_person.TargetFloor -
                                1].append(deepcopy(entering_person))
            self._load_weight += entering_person.Weight
            self.press_button(entering_person.TargetFloor)

        # add passengers to exiting queue
        if(self.is_stopped and self.is_fully_open):
            if(abs(delta_distance) < EPSILON):
                if(len(self._loaded_person[floor_idx]) > 0):
                    if(len(self._exiting_person) < self._config._mpee_number):
                        tmp_unload_person = self._loaded_person[floor_idx].pop(
                            0)
                        self._exiting_person.append(
                            [tmp_unload_person, self._person_entering_time])
                        self._config.log_debug(
                            "Person %s is walking out of the %s elevator",
                            tmp_unload_person,
                            self._name)

        if(len(self._entering_person) > 0):
            self._is_entering = True
        else:
            self._is_entering = False

        if(len(self._exiting_person) > 0 or len(self._loaded_person[floor_idx]) > 0):
            self._is_unloading = True
        else:
            self._is_unloading = False

        # always try closing the door
        if(self.is_fully_open):
            self.require_door_closing()

        # check to make sure everything is working in normal state
        self._check_abnormal_state()

        self._config.log_debug(self.__repr__())

        return energy_consumption, person_statistics, sum(
            len(row) for row in self._loaded_person)

    def require_door_opening(self):
        """
        Requires the door to close for the elevator
        """
        if(self.is_stopped and self._door_open_rate < 1.0 - EPSILON
                and self._is_overloaded_alarm < EPSILON):
            self._is_door_opening = True
            self._keep_door_open_left = self._keep_door_open_lag
            self._is_door_closing = False

    def require_door_closing(self):
        """
        Requires the door to open for the elevator
        """
        if(self._door_open_rate > EPSILON and not self._is_door_opening
                and not self._is_unloading and not self._is_entering
                and self._keep_door_open_left < EPSILON):
            self._is_door_closing = True
            self._config.log_debug("Require door closing succeed")
        else:
            self._config.log_debug(
                "Require door closing failed, door_open_rate = %f, is_unloading = %d, entering_person = %s",
                self._door_open_rate,
                self._is_unloading,
                self._entering_person)

    def person_request_in(self, person):
        """
        Load a person onto the elevator
        Args:
          Person Tuple
        Returns:
          True - if Success
          False - if Overloaded
        """
        assert isinstance(person, PersonType)
        cur_floor = self._current_position / self._floor_height + 1.0
        if(abs(person.SourceFloor - cur_floor) > EPSILON or
                abs(self._current_velocity) > EPSILON or
                abs(self._door_open_rate < 1.0 - 2.0 * EPSILON)):
            self._config.log_debug(
                "Refuse to accommadate person: elevator: %s, person: %s, illegal request",
                self,
                person)
            return False
        if(len(self._entering_person) >= self._mpee_number):
            self._config.log_debug(
                "Refuse to accommadate person: elevator: %s, person: %s, maximum parallel stream arrived",
                self,
                person)
            return False
        # if(self._is_overloaded_alarm > EPSILON):
        #   self._config.log_debug("Refuse to accommadate person: elevator: %s, person: %s, over_loaded", self, person)
        #   self.require_door_closing()
        #   return False
        # add all expected weight
        expected_weight = self._load_weight
        for iter_person, time in self._entering_person:
            expected_weight += iter_person.Weight
        if(expected_weight + person.Weight > self._maximum_capacity):
            self._is_overloaded_alarm = 2.0
            self.require_door_closing()
            return False
        self._entering_person.append(
            [deepcopy(person), self._person_entering_time])
        return True

    def press_button(self, button):
        """
        press a button in the elevator, might be valid or not valid
        Args:
          button: the button to be clicked
        Returns:
          None
        """
        if(button > 0 and button <= self._number_of_floors and button not in self._clicked_buttons):
            self._clicked_buttons.add(button)

    def set_action(self, action):
        """
        Impose standard actions
        """
        assert isinstance(action, ElevatorAction)
        assert isinstance(action.TargetFloor, int)
        assert isinstance(action.DirectionIndicator, int)
        assert (action.TargetFloor >= -1 and action.TargetFloor <= self._number_of_floors)
        assert (action.DirectionIndicator in [-1, 0, 1])
        if(action.TargetFloor >= 0 and action.TargetFloor <= self._config.number_of_floors):
            self._dispatch_target = action.TargetFloor
            self._dispatch_target_direction = action.DirectionIndicator
