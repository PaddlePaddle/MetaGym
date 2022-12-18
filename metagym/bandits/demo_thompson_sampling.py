import gym
import numpy
from numpy import random
import metagym.bandits

class ThompsonSampling(object):
    def __init__(self, arms):
        self.K = arms
        self.reset()

    def reset(self):
        self.record = numpy.zeros((self.K, 2)) + 1

    def update(self, idx, r):
        self.record[idx, 0] += r
        self.record[idx, 1] += 1.0 - r

    def select(self):
        val = random.beta(self.record[:,0], self.record[:,1])
        return numpy.argmax(val)

if __name__=="__main__":
    K=10
    env = gym.make("bandits-v0", arms=K)
    policy = ThompsonSampling(K)
    for i in range(10):
        task = env.sample_task()
        env.set_task(task)
        env.reset()
        policy.reset()
        total_r = 0
        done = False
        while not done:
            idx = policy.select()
            _, r, done, _ = env.step(idx)
            policy.update(idx, r)
            total_r += r
        print("%d th try, Thompson Sampling get %f, expected upper bound %f"%(i, total_r, env.expected_upperbound()))
