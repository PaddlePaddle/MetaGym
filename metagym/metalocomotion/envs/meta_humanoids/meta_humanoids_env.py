import os
import random
import time
from metagym.metalocomotion.envs.utils.walker_base_env import WalkerBaseEnv
from metagym.metalocomotion.envs.meta_humanoids.humanoids import Humanoid

class MetaHumanoidEnv(WalkerBaseEnv):
    def __init__(self):
        self.tasks_been_set = False
        all_config = os.listdir(os.path.join(os.path.dirname(__file__), "..", "assets", "humanoids"))
        self.tra_tasks = list()
        self.tst_tasks = list()
        self.ood_tasks = list()
        for file in all_config:
            if(file.find("humanoid_var_tra") == 0):
                self.tra_tasks.append(file)
            if(file.find("humanoid_var_tst") == 0):
                self.tst_tasks.append(file)
            if(file.find("humanoid_var_ood") == 0):
                self.ood_tasks.append(file)
        self.electricity_cost = 4.25 * WalkerBaseEnv.electricity_cost
        self.stall_torque_cost = 4.25 * WalkerBaseEnv.stall_torque_cost

    def set_task(self, task_file):
        self.robot = Humanoid(task_file)
        WalkerBaseEnv.__init__(self, self.robot)
        self.tasks_been_set = True

    def sample_task(self, task_type=None):
        if(task_type is None or task_type == "OOD"):
            return random.choice(self.tra_tasks)
        elif(task_type == "TEST"):
            return random.choice(self.tst_tasks)
        elif(task_type == "OOD"):
            return random.choice(self.ood_tasks)
        else:
            raise Exception("Unexpected task_type: %s"%task_type)

if __name__=="__main__":
    env = MetaHumanoidEnv()
    env.set_task(env.sample_task(task_type="TEST"))
    #env.set_task("humanoid_var_tst_000.xml")
    env.render(mode="human")
    env.reset()
    done = False
    step = 0
    while not done:
        state, reward, done, info = (env.step(env.action_space.sample()))
        step += 1
        print(step)
        time.sleep(1)
