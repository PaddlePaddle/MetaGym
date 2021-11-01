# We show a simple example to start Quadrupedal with ETG mode here
import gym
import metagym.quadrupedal
import numpy as np
env = gym.make('quadrupedal-v0',render=1,task="stairstair",ETG=1,ETG_path="ESStair_origin.npz")
observation = env.reset()
for i in range(100):
    action = np.zeros(12)
    next_obs, reward, done, info = env.step(action)
