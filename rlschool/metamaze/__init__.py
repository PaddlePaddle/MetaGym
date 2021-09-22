from gym.envs.registration import register
from rlschool.metamaze.envs import MetaMaze3D

register(
    id='meta-maze-3D-v0',
    entry_point='MetaMaze3D',
)
