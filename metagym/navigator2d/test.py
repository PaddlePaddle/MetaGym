import gym
import metagym.navigator2d

def test(max_iteration):
    env= gym.make("navigator-wr-2D-v0", enable_render=False)
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        env.set_task(env.sample_task())
        env.reset()
        done = False
        sum_reward = 0.0
        while not done:
            action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            sum_reward += r
        print("episode is over, sum reward: %f" % sum_reward)

if __name__=="__main__":
    # running random policies
    test(100)
