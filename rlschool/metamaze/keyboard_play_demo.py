import sys
sys.path.append("/Users/wangfan04/Codes/WorldEditors/RLSchool")
from rlschool import make_env

if __name__=='__main__':
    maze_env = make_env("MetaMaze")
    cell_scale = 9
    task = maze_env.sample_task(cell_scale=cell_scale, cell_size=5.0, wall_height=6.4)
    maze_env.set_task(task)
    while True:
        maze_env.reset()
        done=False
        sum_reward = 0
        while not done:
            maze_env.render()
            state, reward, done, _ = maze_env.step()
            sum_reward += reward
        if(not maze_env.key_done):
            print("Episode is over! You got %.1f score"%sum_reward)
            if(sum_reward > 0.0):
                cell_scale += 2 # gradually increase the difficulty
            task = maze_env.sample_task(cell_scale=cell_scale, cell_size=5.0, wall_height=6.4)
            maze_env.set_task(task)
        else:
            break
