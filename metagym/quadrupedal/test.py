# We show a simple example to start Quadrupedal here
import gym
import metagym.quadrupedal
import numpy as np
env = gym.make('quadrupedal-v0',render=1,task="stairstair")
observation = env.reset()
for i in range(100):
    action = np.random.uniform(-0.3,0.3,size=12)
    next_obs, reward, done, info = env.step(action)
