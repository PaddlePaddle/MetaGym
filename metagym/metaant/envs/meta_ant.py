from gym import utils
from meta_mujoco import MetaMujocoEnv
import numpy
from numpy import random
import xml.etree.ElementTree as ET

def gen_pattern_ant():
    return random.random(size=(12,)) * 1.0 + 0.5

def reconfig_xml(file_name, pattern):
    tree = ET.parse(file_name)
    root = tree.getroot()
    wbody = root.find('worldbody')
    body = wbody.find('body')
    idx = 0
    for subbody in body.findall('body'):
        subsubbody = subbody.find("body")
        subsubsubbody = subsubbody.find("body")
        l1 = numpy.asarray(list(map(float, subbody.find("geom").get("fromto").split())))
        l2 = numpy.asarray(list(map(float, subsubbody.find("geom").get("fromto").split())))
        l3 = numpy.asarray(list(map(float, subsubsubbody.find("geom").get("fromto").split())))
        l1 *= pattern[idx]
        l2 *= pattern[idx + 1]
        l3 *= pattern[idx + 2]
        idx += 3
        subbody.find("geom").set("fromto", " ".join(map(str, l1)))
        subsubbody.find("geom").set("fromto", " ".join(map(str, l2)))
        subsubsubbody.find("geom").set("fromto", " ".join(map(str, l3)))
    return ET.tostring(root, encoding='unicode')

class MetaAntEnv(MetaMujocoEnv, utils.EzPickle):
    def __init__(self):
        MetaMujocoEnv.__init__(self, 5)
        utils.EzPickle.__init__(self)
        self.tasks_been_set = False

    def set_task(self, task_config):
        super(MetaMujocoEnv, self).set_tasks(task_config)
        self.tasks_been_set = True

    def sample_task(self):
        return reconfig_xml("img/ant.xml", gen_pattern_ant())

    def reset(self):
        if(not self.tasks_been_set):
            raise Exception("must call set_task before reset")

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

if __name__=="__main__":
    env = MetaAntEnv()
    env.set_task(env.sample_task())
    env.reset()
    done = False
    while not done:
        observation, reward, done, info = env.step(env.action_space.sample())
