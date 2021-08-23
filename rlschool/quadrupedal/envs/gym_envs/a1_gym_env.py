# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
import gym

from rlschool.quadrupedal.envs import env_builder
from rlschool.quadrupedal.robots import a1
from rlschool.quadrupedal.robots import robot_config

SENSOR_MODE = {"dis":1,"motor":1,"imu":1,"contact":1,"footpose":1,"CPG":0}

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
               task = "plane",
               motor_control_mode = robot_config.MotorControlMode.POSITION,
               dynamic_param={'control_latency':0,'footfriction':1,'basemass':1},
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
    self.gait = gait
    self.observation_space = self._env.observation_space
    self.action_space = self._env.action_space
    self.sensor_mode =sensor_mode
    # print('env para:',param)

  def step(self, action):
    # print("gym_act:",action)
    # print("action_scale",self._env.action_space.high)
    # action
    return self._env.step(action)

  def reset(self,**kwargs):
    # print(kwargs)
    # if "stepheight" in kwargs:
    #   return self._env.reset(stepheight=kwargs["stepheight"])
    # else:
    return self._env.reset(**kwargs)

  def close(self):
    self._env.close()

  def render(self, mode):
    return self._env.render(mode) 

  def __getattr__(self, attr):
    return getattr(self._env, attr)