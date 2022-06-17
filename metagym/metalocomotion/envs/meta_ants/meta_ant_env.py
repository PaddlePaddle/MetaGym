import os
import random
from metagym.metalocomotion.envs.utils.walker_base_env import WalkerBaseEnv
from metagym.metalocomotion.envs.meta_ants.ant import Ant

class MetaAntEnv(WalkerBaseEnv):
    def __init__(self):
        self.tasks_been_set = False
        all_config = os.listdir(os.path.join(os.path.dirname(__file__), "..", "assets", "ants"))
        self.tra_tasks = list()
        self.tst_tasks = list()
        self.ood_tasks = list()
        for file in all_config:
            if(file.find("ant_var_tra") == 0):
                self.tra_tasks.append(file)
            if(file.find("ant_var_tst") == 0):
                self.tst_tasks.append(file)
            if(file.find("ant_var_ood") == 0):
                self.ood_tasks.append(file)

    def set_task(self, task_file, render=False, max_steps=2000):
        self.robot = Ant(task_file)
        WalkerBaseEnv.__init__(self, self.robot, render=render, max_steps=max_steps)
        self.tasks_been_set = True

    def sample_task(self, task_type=None):
        if(task_type is None or task_type == "TRAIN"):
            return random.choice(self.tra_tasks)
        elif(task_type == "TEST"):
            return random.choice(self.tst_tasks)
        elif(task_type == "OOD"):
            return random.choice(self.ood_tasks)
        else:
            raise Exception("Unexpected task_type: %s"%task_type)
