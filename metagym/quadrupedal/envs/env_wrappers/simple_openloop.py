# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""Simple openloop trajectory generators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import attr
from gym import spaces
import numpy as np

from metagym.quadrupedal.robots import laikago_pose_utils
from metagym.quadrupedal.robots import minitaur_pose_utils

class MinitaurPoseOffsetGenerator(object):
  """A trajectory generator that return a constant leg pose."""

  def __init__(self,
               init_swing=0,
               init_extension=2.0,
               init_pose=None,
               action_scale=1.0,
               action_limit=0.5):
    """Initializes the controller.

    Args:
      init_swing: the swing of the default pose offset
      init_extension: the extension of the default pose offset
      init_pose: the default pose offset, which is None by default. If not None,
        it will define the default pose offset while ignoring init_swing and
        init_extension.
      action_scale: changes the magnitudes of actions
      action_limit: clips actions
    """
    if init_pose is None:
      self._pose = np.array(
          attr.astuple(
              minitaur_pose_utils.MinitaurPose(
                  swing_angle_0=init_swing,
                  swing_angle_1=init_swing,
                  swing_angle_2=init_swing,
                  swing_angle_3=init_swing,
                  extension_angle_0=init_extension,
                  extension_angle_1=init_extension,
                  extension_angle_2=init_extension,
                  extension_angle_3=init_extension)))
    else:  # Ignore init_swing and init_extension
      self._pose = np.array(init_pose)
    
    action_high = np.array([action_limit] * 4)
    self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self._action_scale = action_scale

  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    return minitaur_pose_utils.leg_pose_to_motor_angles(self._pose +
                                                        self._action_scale *
                                                        np.array(input_action))

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation


class LaikagoPoseOffsetGenerator(object):
  """A trajectory generator that return constant motor angles."""
  def __init__(
      self,
      init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE,
      init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE,
      init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE,
      action_limit=0.5,
      action_space = 0,
  ):
    """Initializes the controller.
    Args:
      action_limit: a tuple of [limit_abduction, limit_hip, limit_knee]
    """
    self._pose = np.array(
        attr.astuple(
            laikago_pose_utils.LaikagoPose(abduction_angle_0=init_abduction,
                                           hip_angle_0=init_hip,
                                           knee_angle_0=init_knee,
                                           abduction_angle_1=init_abduction,
                                           hip_angle_1=init_hip,
                                           knee_angle_1=init_knee,
                                           abduction_angle_2=init_abduction,
                                           hip_angle_2=init_hip,
                                           knee_angle_2=init_knee,
                                           abduction_angle_3=init_abduction,
                                           hip_angle_3=init_hip,
                                           knee_angle_3=init_knee)))
    # print('pose:',self._pose)
    # raise NotImplemented 
    # action_high = np.array([0.802,3.288,0.88] * 4)
    # action_low = np.array([-0.802,-1.94,-0.89]*4)
    self.action_mode = action_space
    if action_space ==0:
      action_high = np.array([0.2,0.7,0.7] *4)
      action_low = np.array([-0.2,-0.7,-0.7]*4)
    elif action_space==1:
      action_high = np.array([0.1,0.5,0.4] * 4)
      action_low = np.array([-0.1,-0.3,-0.6]*4) 
    elif action_space==2:
      action_high = np.array([0.1,0.5,0.4,0.1,0.5,0.4,0.1,0.1,0.1,0.1,0.1,0.1])
      action_low = np.array([-0.1,-0.3,-0.6,-0.1,-0.3,-0.6,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1])
    elif action_space==3:
      action_high = np.array([0.1,0.7,0.7,0.1,0.7,0.7,0.1,0.1,0.1,0.1,0.1,0.1])
      action_low = np.array([-0.1,-0.7,-0.7,-0.1,-0.7,-0.7,-0.1,-0.1,-0.1,-0.1,-0.1,-0.1])
      
    # action_high = np.array([action_limit] * 4).reshape(-1,1)
    # print('action_limit:',action_limit)
    # print('action_high:',action_high+self._pose)
    # print('action_low:',action_low+self._pose)
    self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
    # print('action_space_n:',self.action_space.low,self.action_space.high)
    # raise NotImplemented
  def reset(self):
    pass

  def get_action(self, current_time=None, input_action=None):
    """Computes the trajectory according to input time and action.

    Args:
      current_time: The time in gym env since reset.
      input_action: A numpy array. The input leg pose from a NN controller.

    Returns:
      A numpy array. The desired motor angles.
    """
    del current_time
    # print(self._pose)
    if self.action_mode <=1:
      return self._pose + input_action
    elif self.action_mode >= 2:
      # print('input_act:',input_action)
      new_action = np.zeros(12)
      new_action[6:9] = input_action[3:6]
      new_action[9:12] = input_action[:3]
      new_action += input_action
      # print('new_act:',new_action)
      return self._pose + new_action

  def get_observation(self, input_observation):
    """Get the trajectory generator's observation."""

    return input_observation
