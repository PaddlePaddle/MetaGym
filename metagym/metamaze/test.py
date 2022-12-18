#!/usr/bin/env python
# coding=utf8
# File: test.py
import gym
import sys
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler

def test_2d_maze(max_iteration, task_type):
    print("Testing 2D Maze with task type: ", task_type)
    maze_env = gym.make("meta-maze-2D-v0", max_steps=200, enable_render=False, view_grid=1, task_type=task_type)
    n = 9
    task = MazeTaskSampler(n=n, step_reward=-0.01, goal_reward=1.0)
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
        n += 2 # gradually increase the difficulty
        print("Get score %f, Increase the difficulty, n = %d"%(sum_reward, n))
        task = MazeTaskSampler(n=n, step_reward=-0.01, goal_reward=1.0)
        maze_env.set_task(task)

def test_3d_discrete_maze(max_iteration, task_type):
    print("Testing Discrete 3D Maze with task type: ", task_type)
    maze_env = gym.make("meta-maze-discrete-3D-v0", max_steps=200, enable_render=False, task_type=task_type)
    n = 9
    task = MazeTaskSampler(n=n, step_reward=-0.01, goal_reward=1.0)
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
        n += 2 # gradually increase the difficulty
        print("Get score %f, Increase the difficulty, n = %d"%(sum_reward, n))
        task = MazeTaskSampler(n=n, step_reward=-0.01, goal_reward=1.0)
        maze_env.set_task(task)

def test_3d_continuous_maze(max_iteration, task_type):
    print("Testing Continuous 3D Maze with task type: ", task_type)
    maze_env = gym.make("meta-maze-continuous-3D-v0", max_steps=1000, enable_render=False, task_type=task_type)
    n = 9
    task = MazeTaskSampler(n=n, step_reward=-0.001, goal_reward=1.0)
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
        n += 2 # gradually increase the difficulty
        print("Get score %f, Increase the difficulty, n = %d"%(sum_reward, n))
        task = MazeTaskSampler(n=n, step_reward=-0.001, goal_reward=1.0)
        maze_env.set_task(task)

if __name__=="__main__":
    test_2d_maze(10, "ESCAPE")
    test_2d_maze(10, "SURVIVAL")
    test_3d_discrete_maze(10, "ESCAPE")
    test_3d_discrete_maze(10, "SURVIVAL")
    test_3d_continuous_maze(10, "ESCAPE")
    test_3d_continuous_maze(10, "SURVIVAL")
