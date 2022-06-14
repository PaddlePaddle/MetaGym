import gym
import metagym.metalocomotion

test_num_tasks = 24

def test():
    print("Testing Humanoids ...")
    env = gym.make("meta-humanoid-v0")
    for _ in range(test_num_tasks):
        task = env.sample_task(task_type="TRAIN")
        env.set_task(task)
        env.reset()
        step = 0
        done = False
        while step < 1000 and not done:
            obs, r, done, info = env.step(env.action_space.sample())
            step += 1
    print("...Testing Humanoids Finishes")

    print("Testing Ants ...")
    env = gym.make("meta-ant-v0")
    for _ in range(test_num_tasks):
        task = env.sample_task(task_type="TRAIN")
        env.set_task(task)
        env.reset()
        step = 0
        done = False
        while step < 1000 and not done:
            obs, r, done, info = env.step(env.action_space.sample())
            step += 1
    print("...Testing Ants Finishes")

if __name__=="__main__":
    test()
    print("Test Done")
