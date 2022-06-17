import gym
import time
import metagym.metalocomotion

test_num_tasks = 1

def test_humanoid():
    print("Testing Humanoids ...")
    env = gym.make("meta-humanoid-v0")
    task = env.sample_task(task_type="OOD")
    env.set_task(task, render=False, max_steps=20000)
    env.render()
    env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(env.action_space.sample())
        time.sleep(0.05)
    env.close()

def test_ant():
    print("Testing Humanoids ...")
    env = gym.make("meta-ant-v0")
    task = env.sample_task(task_type="OOD")
    env.set_task(task, render=False, max_steps=20000)
    env.render()
    env.reset()
    done = False
    while not done:
        obs, r, done, info = env.step(env.action_space.sample())
        time.sleep(0.05)
    env.close()

if __name__=="__main__":
    #test_ant()
    test_humanoid()
