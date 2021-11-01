"""Set up gym interface for locomotion environments."""

from gym.envs.registration import register
from metagym.quadrupedal.envs import *

register(
    id='quadrupedal-v0',
    entry_point='metagym.quadrupedal:A1GymEnv',
    kwargs={
       "action_limit": (0.75, 0.75, 0.75),
       "render": False,
       "on_rack": False,
       "random_dynamic" :  False,
       "ETG" :  0,
       "ETG_T" :  0.5,
       "ETG_H" :  20,
       "ETG_path" :  "",
       "task" :  "plane",
       "dynamic_param": {},
    }
)
