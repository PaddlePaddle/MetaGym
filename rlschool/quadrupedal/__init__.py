"""Set up gym interface for locomotion environments."""

from gym.envs.registration import register
from rlschool.quadrupedal.envs import *

register(
    id='quadrupedal-v0',
    entry_point='rlschool.quadrupedal:A1GymEnv',
    kwargs={
       "action_limit": (0.75, 0.75, 0.75),
       "render": False,
       "on_rack": False,
       "sensor_mode" :  SENSOR_MODE,
       "gait" :  0,
       "normal" :  0,
       "filter_" :  0,
       "action_space" : 0,
       "random_dynamic" :  False,
       "reward_param" :  Param_Dict,
       "ETG" :  0,
       "ETG_T" :  0.5,
       "ETG_H" :  20,
       "ETG_path" :  "",
       "random_param" :  Random_Param_Dict,
       "vel_d" :  0.6,
       "step_y" :  0.05,
       "task" :  "plane",
       "reward_p" :  1.0,
       "motor_control_mode" :  robot_config.MotorControlMode.POSITION,
       "dynamic_param": {},
    }
)
