# We show a simple example to start Quadrupedal with ETG mode here
from rlschool import make_env
import numpy as np
env = make_env('Quadrupedal',render=1,task="stairstair",ETG=1,ETG_path="ESStair_origin.npz")
observation = env.reset()
for i in range(100):
    action = np.zeros(12)
    next_obs, reward, done, info = env.step(action)