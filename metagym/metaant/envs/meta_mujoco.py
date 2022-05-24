import sys
from gym.envs.mujoco import mujoco_env


class MetaMujocoEnv(gym.Env, mujoco_env.MujocoEnv):
    """Superclass for all MuJoCo environments."""

    def __init__(self, frame_skip):
        self.frame_skip = frame_skip
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def set_task(self, task_config):
        self.model = mujoco_py.load_model_from_xml(task_config)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
