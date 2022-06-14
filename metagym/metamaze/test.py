import gym
import sys
import metagym.metamaze

def test_2d_maze(max_iteration):
    print("Testing 2D Maze...")
    maze_env = gym.make("meta-maze-2D-v0", enable_render=False, view_grid=1)
    cell_scale = 9
    task = maze_env.sample_task(cell_scale=cell_scale)
    maze_env.set_task(task)
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        maze_env.reset()
        done=False
        sum_reward = 0
        while not done:
            state, reward, done, _ = maze_env.step(maze_env.action_space.sample())
            sum_reward += reward
        print("Episode is over! You got %.1f score."%sum_reward)
        if(sum_reward > 0.0):
            cell_scale += 2 # gradually increase the difficulty
            print("Increase the difficulty, cell_scale = %d"%cell_scale)
        task = maze_env.sample_task(cell_scale=cell_scale)
        maze_env.set_task(task)

def test_3d_maze(max_iteration):
    print("Testing 3D Maze...")
    maze_env = gym.make("meta-maze-3D-v0", enable_render=False)
    cell_scale = 9
    task = maze_env.sample_task(cell_scale=cell_scale, cell_size=2.0, wall_height=3.2)
    maze_env.set_task(task)
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        maze_env.reset()
        done=False
        sum_reward = 0
        while not done:
            state, reward, done, _ = maze_env.step(maze_env.action_space.sample())
            sum_reward += reward
        print("Episode is over! You got %.1f score."%sum_reward)
        if(sum_reward > 0.0):
            cell_scale += 2 # gradually increase the difficulty
            print("Increase the difficulty, cell_scale = %d"%cell_scale)
        task = maze_env.sample_task(cell_scale=cell_scale, cell_size=2.0, wall_height=3.2)
        maze_env.set_task(task)

if __name__=="__main__":
    test_2d_maze(100)
    test_3d_maze(100)
