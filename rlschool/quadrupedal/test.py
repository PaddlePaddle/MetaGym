# We show a simple example to start Quadrupedal here
from rlschool import make_env
import numpy as np
env = make_env('Quadrupedal',render=1,task="stairstair")
observation = env.reset()
for i in range(100):
    action = np.random.uniform(-0.3,0.3,size=12)
    next_obs, reward, done, info = env.step(action)