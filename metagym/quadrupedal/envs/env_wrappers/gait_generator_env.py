# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""A wrapped MinitaurGymEnv with a built-in controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import copy
from metagym.quadrupedal.envs.GaitGenerator.Bezier import BezierGait
from metagym.quadrupedal.envs.OpenLoopSM.SpotOL import BezierStepper
from gym import spaces
from metagym.quadrupedal.robots import laikago_pose_utils
import attr

class GaitGeneratorWrapperEnv(object):
  """A wrapped LocomotionGymEnv with a built-in trajectory generator."""

  def __init__(self, gym_env,vel=0.5,gait_mode=1):
    """Initialzes the wrapped env.

    Args:
      gym_env: An instance of LocomotionGymEnv.
      trajectory_generator: A trajectory_generator that can potentially modify
        the action and observation. Typticall generators includes the PMTG and
        openloop signals. Expected to have get_action and get_observation
        interfaces.

    Raises:
      ValueError if the controller does not implement get_action and
      get_observation.

    """
    self._gym_env = gym_env
    self.bz_step = BezierStepper(dt=self._gym_env.env_time_step,StepVelocity=vel)
    self.bzg = BezierGait(dt=self._gym_env.env_time_step)
    self.timesteps = 0
    self.P_yaw = 5
    self.vel = vel
    self.gait_mode = gait_mode
    if self.gait_mode == 1:
      action_high = np.array([1]*13)
      self.action_space = spaces.Box(-action_high,action_high,dtype=np.float32)
      self.old_act = np.zeros(13)
    elif self.gait_mode ==2:
      action_high = np.array([1]*14)
      action_high[-1] = 2
      action_low = np.array([-1]*14)
      action_low[-1] = -0.1
      self.action_space = spaces.Box(action_low,action_high,dtype=np.float32)
      self.old_act = np.zeros(14)
    elif self.gait_mode == 3:
      init_abduction=laikago_pose_utils.LAIKAGO_DEFAULT_ABDUCTION_ANGLE
      init_hip=laikago_pose_utils.LAIKAGO_DEFAULT_HIP_ANGLE
      init_knee=laikago_pose_utils.LAIKAGO_DEFAULT_KNEE_ANGLE
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
      action_high = np.array([0.1,0.5,0.4] * 4)
      action_low = np.array([-0.1,-0.3,-0.6]*4) 
      self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
        

      # print('action_space:',self.action_space.low,self.action_space.high)
    # obs_high = np.concatenate((self.observation_space.high,np.array([-1]*9)),axis=0)
    # obs_low = np.concatenate((self.observation_space.low,np.array([1]*9)),axis=0)
    # self.observation_space = spaces.Box(obs_low,obs_high,dtype=np.float32)
    
    self.alpha_ = 0.7
    self.CD_SCALE = 0.05
    self.RESIDUALS_SCALE = 0.015
    # if self.gait_mode == 2:
    #   self.RESIDUALS_SCALE = 0.1
    #   self.CD_SCALE = 0.1
    # print('obs_space',self.observation_space.low,self.observation_space.high)
    # print('obs:',self.observation_space)
    # self.action
    # print('init!')
    # raise NotImplemented

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)
    
  def reset(self,**kwargs):
    self.timesteps =0 
    self.obs,info = self._gym_env.reset(**kwargs)
    self.bz_step = BezierStepper(dt=self._gym_env.env_time_step,StepVelocity=self.vel)
    self.bzg = BezierGait(dt=self._gym_env.env_time_step)
    # T_bf_ = copy.deepcopy(self._gym_env.robot.GetFootPositionsInBaseFrame())
    T_b0_ = copy.deepcopy(self._gym_env.robot.GetFootPositionsInBaseFrame())
    Tb_d = {}
    Tb_d["FL"]=T_b0_[0,:]
    Tb_d["FR"]=T_b0_[1,:]
    Tb_d["BL"]=T_b0_[2,:]
    Tb_d["BR"]=T_b0_[3,:]
    self.T_b0 = Tb_d
    _,_,StepLength, LateralFraction, YawRate, StepVelocity,ClearanceHeight,_ = self.bz_step.return_bezier_params()
    # control_param = [StepLength, LateralFraction, YawRate, StepVelocity,ClearanceHeight]
    # leg_phase = self.bzg.Phases
    # print(original_observation)
    # print(control_param)
    # print(leg_phase)
    # obs = np.concatenate((self.obs,control_param,leg_phase),axis=0)
    
    # print(T_b0_)
    # self.T_bf = self.Foot2Dict(T_bf_)
    # self.T_b0 = self.Foot2Dict(T_b0_)
    return self.obs,info

  # def Foot2Dict(Tb):
  #   Tb_d = {}
  #   print(Tb)
  #   Tb_d["FL"]=Tb[0,:]
  #   Tb_d["FR"]=Tb[1,:]
  #   Tb_d["BL"]=Tb[2,:]
  #   Tb_d["BR"]=Tb[3,:]
  #   return Tb_d

  def step(self, action):
    """Steps the wrapped environment.

    Args:
      action: Numpy array. The input action from an NN agent.

    Returns:
      The tuple containing the modified observation, the reward, the epsiode end
      indicator.

    Raises:
      ValueError if input action is None.

    """
    # print(action)
    self.timesteps += 1
    if action is None:
      raise ValueError('Action cannot be None')
    pos, orn, StepLength, LateralFraction, YawRate, StepVelocity, ClearanceHeight, PenetrationDepth = self.bz_step.StateMachine(
        )
    # EXP FILTER
    if self.gait_mode !=3:
      action[:13] = self.alpha_ * self.old_act[:13] + (
          1.0 - self.alpha_) * action[:13]
      self.old_act = action

    ClearanceHeight = 0.05

    StepLength = np.clip(StepLength, self.bz_step.StepLength_LIMITS[0],
                            self.bz_step.StepLength_LIMITS[1])
    StepVelocity = np.clip(StepVelocity,
                            self.bz_step.StepVelocity_LIMITS[0],
                            self.bz_step.StepVelocity_LIMITS[1])
    LateralFraction = np.clip(LateralFraction,
                                self.bz_step.LateralFraction_LIMITS[0],
                                self.bz_step.LateralFraction_LIMITS[1])
    YawRate = np.clip(YawRate, self.bz_step.YawRate_LIMITS[0],
                        self.bz_step.YawRate_LIMITS[1])
    ClearanceHeight = np.clip(ClearanceHeight,
                                self.bz_step.ClearanceHeight_LIMITS[0],
                                self.bz_step.ClearanceHeight_LIMITS[1])
    PenetrationDepth = np.clip(PenetrationDepth,
                                self.bz_step.PenetrationDepth_LIMITS[0],
                                self.bz_step.PenetrationDepth_LIMITS[1])
    contacts = copy.deepcopy(self.obs[1]["FootContactSensor"])
    # print("StepLength:",StepLength,"ClearanceHeight:",ClearanceHeight,"PenetrationDepth:",PenetrationDepth)
    # print(StepVelocity,action[-1])
    # ClearanceHeight = 0.08
    if self.gait_mode == 2:
      StepVelocity += action[-1]
    # r,p,yaw = self._gym_env.robot.GetTrueBaseRollPitchYaw()
    # YawRate += -yaw*self.P_yaw
    if self.timesteps > 5:
        T_bf = self.bzg.GenerateTrajectoryX(StepLength, LateralFraction,
                                            YawRate, StepVelocity, self.T_b0, ClearanceHeight,
                                            PenetrationDepth, contacts)
    else:
        T_bf = self.bzg.GenerateTrajectoryX(0.0, 0.0, 0.0, 1, self.T_b0, ClearanceHeight,
                                            PenetrationDepth, contacts)
        if self.gait_mode!=3:
          action[:] = 0
    if self.gait_mode!=3:
      action[1:] *= self.RESIDUALS_SCALE
    leg_id = 0
    action_ref = np.zeros(12)
    for key in T_bf:
        # action[2+3*leg_id]=1
        if self.gait_mode!=3:
          leg_pos = T_bf[key]+ action[1+3*leg_id:4+3*leg_id]
        else:
          leg_pos = T_bf[key]
        # print(T_bf[key],leg_pos)
        index, angle = self._gym_env.robot.ComputeMotorAnglesFromFootLocalPosition(leg_id,leg_pos)
        action_ref[index] = np.asarray(angle)
        # print('leg:{}:'.format(leg_id),angle)
        leg_id += 1
    new_action =action_ref
    if self.gait_mode!=3:
      self.obs, reward, done, info = self._gym_env.step(new_action)
    else:
      new_action = action_ref + action
      self.obs,reward,done,info = self._gym_env.step(new_action)
      info['ref_action'] = action_ref 
      info['ref_leg'] = T_bf
      
    # control_param = [StepLength, LateralFraction, YawRate, StepVelocity,ClearanceHeight]
    # leg_phase = self.bzg.Phases
    # # print(original_observation)
    # # print(control_param)
    # # print(leg_phase)
    # obs = np.concatenate((original_observation,control_param,leg_phase),axis=0)
    
    # print(obs)
    # print('step_vel：',StepVelocity)
    info['real_action'] = new_action

    return self.obs, reward, done, info
