import os
import sys

sys.path.append(os.path.dirname(__file__))

import gym


class Scene(object):
    """A base class for single- and multiplayer scenes"""

    def __init__(self, bullet_client, gravity, timestep, frame_skip):
        self._p = bullet_client
        self.gravity = gravity
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.numSolverIterations = 5
        self.np_random, seed = gym.utils.seeding.np_random(None)
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.dt = self.timestep * self.frame_skip

        self.test_window_still_open = True  # or never opened
        self.human_render_detected = False  # if user wants render("human"), we open test window

        self.multiplayer_robots = {}

    def test_window(self):
        "Call this function every frame, to see what's going on. Not necessary in learning."
        self.human_render_detected = True
        return self.test_window_still_open

    def actor_introduce(self, robot):
        "Usually after scene reset"
        if not self.multiplayer: return
        self.multiplayer_robots[robot.player_n] = robot

    def actor_is_active(self, robot):
        """
        Used by robots to see if they are free to exclusiveley put their HUD on the test window.
        Later can be used for click-focus robots.
        """
        return not self.multiplayer

    def global_step(self):
        """
        The idea is: apply motor torques for all robots, then call global_step(), then collect
        observations from robots using step() with the same action.
        """
        self._p.stepSimulation()

    def reset(self):
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -self.gravity)
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(fixedTimeStep=self.timestep*self.frame_skip, numSolverIterations=self.numSolverIterations, numSubSteps=self.frame_skip)
