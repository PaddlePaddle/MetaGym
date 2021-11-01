# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
import gym

from metagym.quadrupedal.envs import env_builder
from metagym.quadrupedal.robots import a1
from metagym.quadrupedal.robots import robot_config
from metagym.quadrupedal.envs.env_wrappers.MonitorEnv import EnvWrapper,Param_Dict,Random_Param_Dict
from copy import copy

SENSOR_MODE = {"dis":1,"motor":1,"imu":1,"contact":1,"footpose":0,"ETG":0}

class A1GymEnv(gym.Env):
  """A1 environment that supports the gym interface."""
  metadata = {'render.modes': ['rgb_array']}

  def __init__(self,
               action_limit=(0.75, 0.75, 0.75),
               render=False,
               on_rack=False,
               sensor_mode = SENSOR_MODE,
               gait = 0,
               normal = 0,
               filter_ = 0,
               action_space =0,
               random_dynamic = False,
               reward_param = Param_Dict,
               ETG = 0,
               ETG_T = 0.5,
               ETG_H = 20,
               ETG_path = "",
               random_param = Random_Param_Dict,
               vel_d = 0.6,
               step_y = 0.05,
               task = "plane",
               reward_p = 1.0,
               motor_control_mode = robot_config.MotorControlMode.POSITION,
               dynamic_param={},
               **kwargs):
    self._env = env_builder.build_regular_env(
        a1.A1,
        motor_control_mode=motor_control_mode,
        gait = gait,
        normal=normal,
        task_mode=task,
        enable_rendering=render,
        action_limit=action_limit,
        sensor_mode = sensor_mode,
        random=random_dynamic,
        filter = filter_,
        action_space = action_space,
        on_rack=on_rack,
        param = dynamic_param)

    self._env = EnvWrapper(
        env = self._env,
        param = reward_param,
        sensor_mode = sensor_mode,
        normal = normal,
        ETG_T = ETG_T,
        ETG_path = ETG_path,
        ETG = ETG,
        random_param = random_param,
        vel_d = vel_d,
        step_y = step_y,
        task_mode = task,
        reward_p = reward_p,
    )
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.sensor_mode =sensor_mode

  def step(self, action,**kwargs):
    return self._env.step(action,**kwargs)

  def reset(self,**kwargs):
    return self._env.reset(**kwargs)

  def close(self):
    self._env.close()

  def render(self, mode="human"):
    if(mode != "human"):
        raise NotImplementedError("Only human mode is supported")
    return self._env.render(mode) 

  def __getattr__(self, attr):
    return getattr(self._env, attr)
