import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import pybullet
from pybullet_utils import bullet_client

from pkg_resources import parse_version


class BaseBulletEnv(gym.Env):
    """
    Base class for Bullet physics simulation environments in a Scene.
    These environments create single-player scenes and behave like normal Gym environments, if
    you don't use multiplayer.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
        }

    def __init__(self, frame_skip=4, time_step=0.005, render=False):
        self.scene = None
        self.isRender = render
        self._cam_dist = 3
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240

        self.frame_skip = frame_skip
        self.time_step = time_step

        self._robot_set = False
        self._scene_set = False

        if self.isRender:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

    def set_robot(self, robot):
        self.robot = robot
        self.action_space = robot.action_space
        self.observation_space = robot.observation_space
        self._robot_set = True

    def set_scene(self, scene_type):
        self.scene = scene_type(self._p, 9.8, self.time_step, self.frame_skip)
        self._scene_set = True

    def configure(self, args):
        if(self._robot_set):
            self.robot.args = args
        else:
            raise Exception("BaseBulletEnv::configure: must call set_robot first")

    def _seed(self, seed=None):
        if(self._robot_set):
            self.np_random, seed = gym.utils.seeding.np_random(seed)
            self.robot.np_random = self.np_random # use the same np_randomizer for robot as for env
        else:
            raise Exception("BaseBulletEnv::_seed: must call set_robot first")
        return [seed]

    def reset(self):
        self._seed()
        if(not self._robot_set or not self._scene_set):
            raise Exception("BaseBulletEnv::_reset: must call set_robot and set_scene first")

        self.scene.reset()
        self.robot.scene = self.scene

        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0)
        obs = self.robot.reset(self._p)

        self.frame = 0
        self.done = 0
        self.reward = 0
        self.potential = self.robot.calc_potential()

        return obs

    def render(self, mode, close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        base_pos = [0, 0, 0]
        if hasattr(self,'robot'):
            if hasattr(self.robot,'body_xyz'):
                base_pos = self.robot.body_xyz

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(self._render_width)/self._render_height,
            nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = self._p.getCameraImage(
            width = self._render_width, 
            height=self._render_height, 
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
            )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array

    def close(self):
        self._p.resetSimulation()
        self._p.disconnect()

    def step(self, action):
        self.robot.apply_action(action)
        self.scene.global_step()
