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

# mansion_config.py
# A basic class that stores the static information
#
import sys
import time
from rlschool.liftsim.environment.mansion.utils import SimulatedTime
from rlschool.liftsim.environment.mansion.utils import formulate_simulation_time, simulation_time_to_str, raw_time_to_str
from rlschool.liftsim.environment.mansion.utils import args_to_string


class MansionConfig(object):
    def __init__(self,
                 start_time=0.0,
                 dt=0.10,
                 number_of_floors=30,
                 floor_height=4.0,
                 maximum_acceleration=1.0,
                 maximum_speed=2.0,
                 person_entering_time=2.0,
                 door_opening_closing_time=2.0,
                 automatic_door_power=350,
                 standby_power_consumption=100,
                 maximum_capacity=1600,
                 maximum_parallel_entering_exiting_number=2,
                 rated_load=600,
                 net_weight=300,
                 pulley_radius=0.27,
                 motor_efficiency=0.8,
                 motor_gear_ratio=1.0,
                 gear_efficiency=1.0,
                 gear_radius = 1.0):

        self._start_time = start_time
        self._current_time = start_time
        self._delta_t = dt
        # Logger Level: 1. Debug, 2. Notice, 3. Warning
        self._logger_level = 2
        self._logger_stream_std = sys.stdout
        self._logger_stream_err = sys.stderr
        self._cache_up_to_date = False
        self._cache_simulate_time = SimulatedTime(0, 0, 0, 0)
        self._stdoutfile = None
        self._stderrfile = None
        self._add_simulation_time_in_log = True
        self._person_entering_time = person_entering_time
        self._mpee_number = maximum_parallel_entering_exiting_number

        self.number_of_floors = number_of_floors
        self.floor_height = floor_height
        self.maximum_acceleration = maximum_acceleration
        self.maximum_speed = maximum_speed
        self.door_opening_closing_time = door_opening_closing_time
        self.automatic_door_power = automatic_door_power
        self.standby_power_consumption = standby_power_consumption
        self.maximum_capacity = maximum_capacity
        self.rated_load = rated_load
        self.net_weight = net_weight
        self.pulley_radius = pulley_radius
        self.motor_efficiency = motor_efficiency
        self.motor_gear_ratio = motor_gear_ratio
        self.gear_efficiency = gear_efficiency
        self.gear_radius = gear_radius

    def set_logger_level(self, level):
        if(level.find("Debug") >= 0):
            self._logger_level = 1
        elif(level.find("Notice") >= 0):
            self._logger_level = 2
        elif(level.find("Warning") >= 0):
            self._logger_level = 3
        else:
            raise Exception("No such log level: %s" % level)

    def set_std_logfile(self, filename):
        if(filename.strip() == self._stderrfile):
            self._logger_stream_std = self._logger_stream_err
        else:
            self._logger_stream_std = open(filename, 'w')
        self._stdoutfile = filename

    def set_err_logfile(self, filename):
        if(filename.strip() == self._stdoutfile):
            self._logger_stream_err = self._logger_stream_std
        else:
            self._logger_stream_err = open(filename, 'w')
        self._stderrfile = filename

    def add_simulation_time_in_log(self):
        self._add_simulation_time_in_log = True

    def remove_simulation_time_in_log(self):
        self._add_simulation_time_in_log = False

    def step(self):
        self._current_time += self._delta_t
        self._cache_up_to_date = False

    @property
    def raw_time(self):
        return self._current_time

    @property
    def world_time(self):
        if(not self._cache_up_to_date):
            self._cache_simulate_time = formulate_simulation_time(
                self._current_time)
        return self._cache_simulate_time

    @property
    def delta_t(self):
        return self._delta_t

    def log_debug(self, string, *args):
        if(self._logger_level < 2):
            self._logger_stream_std.write(
                time.strftime(
                    '%Y-%m-%d\t%H:%M:%S',
                    time.localtime(
                        time.time())))
            self._logger_stream_std.write("\tDebug\t")
            if(self._add_simulation_time_in_log):
                self._logger_stream_std.write(
                    "world_time:%s\t" %
                    simulation_time_to_str(
                        self.world_time))
            self._logger_stream_std.write(args_to_string(string, args))
            self._logger_stream_std.write("\n")

    def log_notice(self, string, *args):
        if(self._logger_level < 3):
            self._logger_stream_std.write(
                time.strftime(
                    '%Y-%m-%d\t%H:%M:%S',
                    time.localtime(
                        time.time())))
            self._logger_stream_std.write("\tNotice\t")
            if(self._add_simulation_time_in_log):
                self._logger_stream_std.write(
                    "world_time:%s\t" %
                    simulation_time_to_str(
                        self.world_time))
            self._logger_stream_std.write(args_to_string(string, args))
            self._logger_stream_std.write("\n")

    def log_warning(self, string, *args):
        self._logger_stream_err.write(
            time.strftime(
                '%Y-%m-%d\t%H:%M:%S',
                time.localtime(
                    time.time())))
        self._logger_stream_err.write("\tWarning\t")
        if(self._add_simulation_time_in_log):
            self._logger_stream_err.write(
                "world_time:%s\t" %
                simulation_time_to_str(
                    self.world_time))
        self._logger_stream_err.write(args_to_string(string, args))
        self._logger_stream_err.write("\n")

    def log_fatal(self, string, *args):
        self._logger_stream_err.write(
            time.strftime(
                '%Y-%m-%d\t%H:%M:%S',
                time.localtime(
                    time.time())))
        self._logger_stream_err.write("\tFatal\t")
        if(self._add_simulation_time_in_log):
            self._logger_stream_err.write(
                "world_time:%s\t" %
                simulation_time_to_str(
                    self.world_time))
        raise RuntimeError(args_to_string(string, args))

    def reset(self):
        self._current_time = self._start_time
