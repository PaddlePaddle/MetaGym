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
"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from rlschool.A1_robot.envs.utilities import pose3d
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
    # rot_quat = env.robot.GetBaseOrientation()
    # rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    # foot_links = env.robot.GetFootLinkIDs()
    # ground = env.get_ground()
    # contact_fall = False
    # # sometimes the robot can be initialized with some ground penetration
    # # so do not check for contacts until after the first env step.
    # if env.env_step_counter > 0:
    #   # print('start!')
    #   robot_ground_contacts = env.pybullet_client.getContactPoints(
    #       bodyA=env.robot.quadruped, bodyB=ground)

    #   for contact in robot_ground_contacts:
    #     # print('contact',contact)
    #     if contact[3] not in foot_links:
    #       contact_fall = True
    #       break
      # print('foot_links',foot_links)
    # rot_mat = np.asarray(rot_mat)
    # print('robot_mat:',rot_mat.reshape(3,3),rot_mat.size)
    # print("rot_mat:",rot_mat[-1])
    # return rot_mat[-1]<0.85
    # return rot_mat[-1] < 0.85 or contact_fall
    return False


  def reward(self, env,action,torques):
    """Get the reward without side effects."""
    # del env
    # print('action:',action)
    # print('torques:',torques)
    vel_reward = self._calc_vel_reward()
    # basepos_reward = self.param['base']*self._calc_basepos_reward(env)
    # torque_reward = self.param['tau']*self._calc_torque_reward(torques)
    # foot_reward = self.param['foot']*self._calc_foot_reward(env)
    # print('foot_reward',foot_reward)
    # print('vel:{} base:{} vel_tem:{}'.format(self.param['vel'],self.param['base'],self.param['vel_tem']))
    return vel_reward
    # return self.current_base_pos[0] - self.last_base_pos[0]
  
  def _calc_torque_reward(self,torques):
    tau = np.sum(np.abs(torques))/120.
    return np.exp(-self.param['tau_tem']*tau)

  def _calc_vel_reward(self):
    # if self.param['vel_tem']!=0:
    #   vel_x =(self.current_base_pos[0] - self.last_base_pos[0])/0.1
    #   if vel_x>=0.8:
    #     return 1
    #   else:
    #     return np.exp(-self.param['vel_tem']*np.power(vel_x-0.8,2))
    # else:
      return self.current_base_pos[0] - self.last_base_pos[0]
  
  def _calc_basepos_reward(self,env):
    rot_quat = env.robot.GetBaseOrientation()
    rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
    rot_mat = np.asarray(rot_mat).reshape(3,3)
    pos_penal = np.exp(-1.5*np.power(np.linalg.norm(rot_mat-np.eye(3,3)),2))
    vel_y = (self.current_base_pos[1] - self.last_base_pos[1])/0.1
    vel_penal = np.exp(-self.param['base_tem']*np.power(vel_y,2))
    return vel_penal

  def _calc_foot_reward(self,env):
    #foot 1 frontleft 2 frontright 3 backleft 4 backright
    robot = env._robot
    pose = env._robot.GetFootPositionsInBaseFrame()
    front_vel = max(pose[0][0]-self.last_foot[0],pose[1][0]-self.last_foot[1])
    back_vel = max(pose[2][0]-self.last_foot[2],pose[3][0]-self.last_foot[3])
    # print('front_vel:{} back_vel:{}'.format(front_vel,ba))
    for i in range(4):
      self.last_foot[i] = pose[i][0]
    # front_err = np.abs(pose[0][0]-pose[1][0])
    # back_err = np.abs(pose[2][0]-pose[3][0])
    # print(pose)
    # print(pose[0][0],pose[1][0],pose[2][0],pose[3][0])
    return front_vel+back_vel
    # print(pose)
