import gym
import time
import metagym.metalocomotion

test_num_tasks = 1

def test_humanoid():
    print("Testing Humanoids ...")
    env = gym.make("meta-humanoid-v0", enable_render=True, max_steps=1000)
    task = env.sample_task(task_type="OOD")
    env.set_task(task)
    env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(env.action_space.sample())
        env.render(mode="rgb_array")
        time.sleep(0.05)
    env.close()

def test_ant():
    print("Testing Ants ...")
    env = gym.make("meta-ant-v0", enable_render=True, max_steps=1000)
    task = env.sample_task(task_type="OOD")
    env.set_task(task)
    env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(env.action_space.sample())
        env.render(mode="rgb_array")
        time.sleep(0.05)
    env.close()

if __name__=="__main__":
    #test_humanoid()
    test_ant()
