import sys
import parl
import numpy as np
import numpy.random as random

from copy import deepcopy
from collections import deque

from environment.mansion.utils import ElevatorState, ElevatorAction, MansionState
from environment.mansion.utils import EPSILON, HUGE
from environment.wrapper_utils import mansion_state_preprocessing, obs_dim, act_dim
from environment.wrapper_utils import action_idx_to_action, action_to_action_idx
from baseline.rl_benchmark.model import RLDispatcherModel, ElevatorAgent
from parl.algorithms import DQN
from parl.utils import ReplayMemory

MEMORY_SIZE = 1000000
BATCH_SIZE = 64


class RL_dispatcher():
    '''
    An RL benchmark for elevator system
    '''

    def __init__(self, env, max_episode):
        self.env = env

        # load_settings( )
        self._obs_dim = env.observation_space
        self._act_dim = env.action_space
        self._global_step = 0
        self.max_episode = max_episode
        self._rpm = ReplayMemory(MEMORY_SIZE, self._obs_dim, 1)
        self._model = RLDispatcherModel(self._act_dim)
        hyperparas = {
            'action_dim': self._act_dim,
            'lr': 5.0e-4,
            'gamma': 0.998
        }
        #print ("action dimention:", self._obs_dim, self._act_dim)
        self._algorithm = DQN(self._model, hyperparas)
        self._agent = ElevatorAgent(
            self._algorithm, self._obs_dim, self._act_dim)
        self._warm_up_size = 2000
        self._statistic_freq = 1000
        self._loss_queue = deque()

    def run_episode(self):
        self.env.reset()
        acc_reward = 0.0

        while self._global_step < self.max_episode:
            state = self.env.state
            action = self.policy(state)
            state_, reward, done, info = self.env.step(action)
            output_info = self.learn_step(state, action, reward)
            acc_reward += reward
            if (isinstance(output_info, dict) and len(output_info) > 0): # TODO:
                self.env.log_notice("%s", output_info)
            if(self._global_step % 3600 == 0):
                self.env.log_notice(
                    "Accumulated Reward: %f, Mansion Status: %s",
                    acc_reward, self.env.statistics)
                acc_reward = 0.0

    def learn_step(self, state, action, r):
        self._global_step += 1
        # observation_array = mansion_state_preprocessing(state)
        # new_actions = list()
        # for ele_act in action:
        #     new_actions.append(action_to_action_idx(ele_act, self._act_dim))
        if(self._global_step > self._warm_up_size):
            for i in range(self.env.elevator_num):
                self._rpm.append(
                    self._last_observation_array[i],
                    self._last_action[i],
                    self._last_reward,
                    deepcopy(state[i]), False)
        self._last_observation_array = deepcopy(state)
        self._last_action = deepcopy(action)
        self._last_reward = r

        ret_dict = {}
        if self._rpm.size() > self._warm_up_size:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = \
                self._rpm.sample_batch(BATCH_SIZE)
            cost = self._agent.learn(
                batch_obs,
                batch_action,
                batch_reward,
                batch_next_obs,
                batch_terminal)
            self._loss_queue.appendleft(cost)
            if(len(self._loss_queue) > self._statistic_freq):
                self._loss_queue.pop()
            if(self._global_step % self._statistic_freq == 0):
                ret_dict["Temporal Difference Error(Average)"] = \
                    float(sum(self._loss_queue)) / float(len(self._loss_queue))

        return ret_dict

    def policy(self, state):
        self._exploration_ratio = 800000.0 / \
            (800000.0 + self._global_step) + 0.1
        if self._global_step > 200000 and self._global_step % 50000 <= 3000:
            self._exploration_ratio = 0
        # observation_array = mansion_state_preprocessing(state)
        q_values = self._agent.predict(state)
        ret_actions = list()
        for i in range(self.env.elevator_num):
            if(random.random() < self._exploration_ratio):
                action = random.randint(0, self._act_dim)
            else:
                action = np.argmax(q_values[i])
            ret_actions.append(int(action))
        return ret_actions
