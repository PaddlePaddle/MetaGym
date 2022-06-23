import gym
import metagym.metalocomotion

def test(test_num_tasks):
    print("Testing Humanoids ...")
    env = gym.make("meta-humanoid-v0", enable_render=False, max_steps=2)
    for _ in range(test_num_tasks):
        task = env.sample_task(task_type="TRAIN")
        env.set_task(task)
        env.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(env.action_space.sample())
    env.close()
    print("...Testing Humanoids Finishes")

    print("Testing Ants ...")
    env = gym.make("meta-ant-v0", enable_render=False, max_steps=2)
    for _ in range(test_num_tasks):
        task = env.sample_task(task_type="TRAIN")
        env.set_task(task)
        env.reset()
        done = False
        while not done:
            obs, r, done, info = env.step(env.action_space.sample())
    env.close()
    print("...Testing Ants Finishes")

if __name__=="__main__":
    for _ in range(100):
        test(100)
