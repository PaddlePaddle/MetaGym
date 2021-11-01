# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""A gin-config class for locomotion_gym_env.

This should be identical to locomotion_gym_config.proto.
"""
import attr
import typing
from metagym.quadrupedal.robots import robot_config


@attr.s
class SimulationParameters(object):
  """Parameters specific for the pyBullet simulation."""
  sim_time_step_s = attr.ib(type=float, default=0.002)
  num_action_repeat = attr.ib(type=int, default=33)
  enable_hard_reset = attr.ib(type=bool, default=False)
  enable_rendering = attr.ib(type=bool, default=False)
  enable_rendering_gui = attr.ib(type=bool, default=True)
  robot_on_rack = attr.ib(type=bool, default=False)
  camera_distance = attr.ib(type=float, default=1.0)
  camera_yaw = attr.ib(type=float, default=0)
  camera_pitch = attr.ib(type=float, default=-30)
  render_width = attr.ib(type=int, default=480)
  render_height = attr.ib(type=int, default=360)
  egl_rendering = attr.ib(type=bool, default=False)
  motor_control_mode = attr.ib(type=int,
                               default=robot_config.MotorControlMode.POSITION)
  reset_time = attr.ib(type=float, default=-1)
  enable_action_filter = attr.ib(type=bool, default=True)
  enable_action_interpolation = attr.ib(type=bool, default=True)
  allow_knee_contact = attr.ib(type=bool, default=False)
  enable_clip_motor_commands = attr.ib(type=bool, default=True)


@attr.s
class ScalarField(object):
  """A named scalar space with bounds."""
  name = attr.ib(type=str)
  upper_bound = attr.ib(type=float)
  lower_bound = attr.ib(type=float)


@attr.s
class LocomotionGymConfig(object):
  """Grouped Config Parameters for LocomotionGym."""
  simulation_parameters = attr.ib(type=SimulationParameters)
  log_path = attr.ib(type=typing.Text, default=None)
  profiling_path = attr.ib(type=typing.Text, default=None)
