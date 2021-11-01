# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation/blob/master/motion_imitation/envs/env_wrappers/simple_forward_task.py

"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from metagym.quadrupedal.envs.utilities import pose3d
from pybullet_utils import transformations

class SimpleForwardTask(object):
  """Default empy task."""
  def __init__(self,param):
    """Initializes the task."""
    self.current_base_pos = np.zeros(3)
    self.last_base_pos = np.zeros(3)
    self.param = param
    self.last_foot = np.zeros(4)

  def __call__(self, env,action,torques):
    return self.reward(env,action,torques)

  def reset(self, env):
    """Resets the internal state of the task."""
    self._env = env
    self.last_base_pos = env.robot.GetBasePosition()
    self.current_base_pos = self.last_base_pos

  def update(self, env):
    """Updates the internal state of the task."""
    self.last_base_pos = self.current_base_pos
    self.current_base_pos = env.robot.GetBasePosition()

  def done(self, env):
    """Checks if the episode is over.

       If the robot base becomes unstable (based on orientation), the episode
       terminates early.
    """
    return False


  def reward(self, env,action,torques):
    """Get the reward without side effects."""
    vel_reward = self._calc_vel_reward()
    return vel_reward
  
  def _calc_vel_reward(self):
      return self.current_base_pos[0] - self.last_base_pos[0]
  

