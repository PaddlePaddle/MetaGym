# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from rlschool.A1_robot.envs import locomotion_gym_env
from rlschool.A1_robot.envs import locomotion_gym_config
from rlschool.A1_robot.envs.env_wrappers import observation_dictionary_to_array_wrapper as obs_dict_to_array_wrapper
from rlschool.A1_robot.envs.env_wrappers import trajectory_generator_wrapper_env
from rlschool.A1_robot.envs.env_wrappers import simple_openloop
from rlschool.A1_robot.envs.env_wrappers import simple_forward_task
from rlschool.A1_robot.envs.sensors import robot_sensors
from rlschool.A1_robot.robots import a1
from rlschool.A1_robot.robots import robot_config
from rlschool.A1_robot.envs.env_wrappers.gait_generator_env import GaitGeneratorWrapperEnv




def build_regular_env(robot_class,
                      motor_control_mode,
                      param,
                      sensor_mode,
                      gait = 0,
                      enable_rendering=False,
                      on_rack=False,
                      filter = 0,
                      action_space = 0,
                      action_limit=(0.75, 0.75, 0.75),
                      wrap_trajectory_generator=True):

  sim_params = locomotion_gym_config.SimulationParameters()
  sim_params.enable_rendering = enable_rendering
  sim_params.motor_control_mode = motor_control_mode
  sim_params.reset_time = 2
  sim_params.num_action_repeat = 10
  sim_params.enable_action_interpolation = False
  if filter:
    sim_params.enable_action_filter = True
  else:
    sim_params.enable_action_filter = False
  sim_params.enable_clip_motor_commands = False
  sim_params.robot_on_rack = on_rack
#   sim_params.sim_time_step_s = 1./500.

  gym_config = locomotion_gym_config.LocomotionGymConfig(
      simulation_parameters=sim_params)
  sensors = []
  if sensor_mode["dis"]:
    sensors.append(robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True))
  if sensor_mode["imu"]:
    sensors.append(robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"]))
  if sensor_mode["motor"]:
    sensors.append(robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS))
  if sensor_mode["contact"]:
    sensors.append(robot_sensors.FootContactSensor())

  if sensor_mode == 0:
    sensors = [
        # robot_sensors.BasePositionSensor(),
        robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True),
        robot_sensors.IMUSensor(channels=["R", "P", "Y"]),
        # robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
        #   robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS),
        #   robot_sensors.FootForceSensor()
    ]
  elif sensor_mode == 1:
    sensors = [
        robot_sensors.BaseDisplacementSensor(),
        robot_sensors.IMUSensor(),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.FootForceSensor()
    ]
  elif sensor_mode == 2:
    sensors = [
        robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True),
        robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"]),
        robot_sensors.MotorAngleSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.SimpleFootForceSensor()
    ]
  elif sensor_mode >= 3 and sensor_mode<=6:
    sensors = [
        robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True),
        robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"]),
        robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.SimpleFootForceSensor()
    ]
  elif sensor_mode >=8:
    sensors = [
        robot_sensors.BaseDisplacementSensor(convert_to_local_frame=True),
        robot_sensors.IMUSensor(channels=["R", "P", "Y","dR", "dP", "dY"]),
        robot_sensors.MotorAngleAccSensor(num_motors=a1.NUM_MOTORS),
        robot_sensors.FootContactSensor()
    ]



  task = simple_forward_task.SimpleForwardTask(param)

  env = locomotion_gym_env.LocomotionGymEnv(gym_config=gym_config,
                                            param = param,
                                            robot_class=robot_class,
                                            robot_sensors=sensors,
                                            task=task)

  env = obs_dict_to_array_wrapper.ObservationDictionaryToArrayWrapper(
      env)
  if gait!=0 and (motor_control_mode
      == robot_config.MotorControlMode.POSITION):
    env = GaitGeneratorWrapperEnv(env,gait_mode=gait)
  elif (motor_control_mode
      == robot_config.MotorControlMode.POSITION) and wrap_trajectory_generator:
    # if robot_class == laikago.Laikago:
    #   env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
    #       env,
    #       trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
    #           action_limit=0.75)) #origin action_limit=action_limit
        
    # elif robot_class == a1.A1:
    env = trajectory_generator_wrapper_env.TrajectoryGeneratorWrapperEnv(
        env,
        trajectory_generator=simple_openloop.LaikagoPoseOffsetGenerator(
            action_limit=0.75,action_space=action_space)) #origin action_limit=action_limit

  return env