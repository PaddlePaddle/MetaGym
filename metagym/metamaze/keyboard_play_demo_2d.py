import gym
import sys
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler

if __name__=='__main__':
    maze_env = gym.make("meta-maze-2D-v0", max_steps=1000, view_grid=1, task_type="ESCAPE")
    n = 15
    task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
    maze_env.set_task(task)
    while True:
        maze_env.reset()
        done=False
        sum_reward = 0
        while not done:
            maze_env.render()
            state, reward, done, _ = maze_env.step(None)
            sum_reward += reward
        if(not maze_env.key_done):
            print("Episode is over! You got %.1f score."%sum_reward)
            maze_env.save_trajectory("trajectory_%dx%d.png"%(n, n))
            if(sum_reward > 0.0):
                n += 2 # gradually increase the difficulty
                print("Increase the difficulty, n = %d"%n)
            task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
            maze_env.set_task(task)
        else:
            break
