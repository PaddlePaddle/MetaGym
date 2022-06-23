import os
import random
from metagym.metalocomotion.envs.utils.walker_base_env import WalkerBaseEnv
from metagym.metalocomotion.envs.utils.stadium import StadiumScene
from metagym.metalocomotion.envs.meta_humanoids.humanoids import Humanoid

class MetaHumanoidEnv(WalkerBaseEnv):
    def __init__(self, frame_skip=4, time_step=0.005, enable_render=False, max_steps=2000):
        super(MetaHumanoidEnv, self).__init__(
                frame_skip=frame_skip,
                time_step=time_step,
                render=enable_render,
                max_steps=max_steps
                )
        self.set_scene(StadiumScene)

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

    def set_task(self, task_file):
        self.set_robot(Humanoid(task_file))

    def sample_task(self, task_type=None):
        if(task_type is None or task_type == "TRAIN"):
            return random.choice(self.tra_tasks)
        elif(task_type == "TEST"):
            return random.choice(self.tst_tasks)
        elif(task_type == "OOD"):
            return random.choice(self.ood_tasks)
        else:
            raise Exception("Unexpected task_type: %s"%task_type)
