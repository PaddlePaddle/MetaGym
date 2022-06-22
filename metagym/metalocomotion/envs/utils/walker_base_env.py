from .env_bases import BaseBulletEnv
from .stadium import StadiumScene
import pybullet as p
import numpy as np

class WalkerBaseEnv(BaseBulletEnv):
    def __init__(self, frame_skip=4, time_step=0.005, render=False, max_steps=2000):
        super(WalkerBaseEnv, self).__init__(
                frame_skip=frame_skip,
                time_step=time_step,
                render=render
                )
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.stateId=-1
        self.max_steps = max_steps

        self.electricity_cost     = 0.0#-2.0     # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
        self.stall_torque_cost    = 0.0#-0.1     # cost for running electric current through a motor even at zero rotational speed, small
        self.foot_collision_cost  = -1.0     # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
        self.foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
        self.joints_at_limit_cost = -0.1     # discourage stuck joints

    def reset(self):
        obs = super(WalkerBaseEnv, self).reset()

        self.steps = 0
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.robot.addToScene(self._p,
                self.scene.ground_plane_mjcf)
        self.ground_ids = set([(self.parts[f].bodies[self.parts[f].bodyIndex], self.parts[f].bodyPartIndex) for f in
                self.foot_ground_object_names])
        self._p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        #if self.stateId >= 0:
        #    self._p.restoreState(self.stateId)
        #else:
        #    self.stateId=self._p.saveState()

        return obs

    def step(self, action):
        super(WalkerBaseEnv, self).step(action)

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("...state suffer from too large number...", state)
            done = True

        potential_old = self.potential
        self.potential = self.robot.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet):
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            if self.ground_ids & contact_ids:
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0

        electricity_cost  = self.electricity_cost  * float(np.abs(action * self.robot.joint_speeds).mean()) 
        electricity_cost += self.stall_torque_cost * float(np.square(action).mean())
        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        inst_reward = sum(rewards)
        self.reward += inst_reward 
        self.steps += 1
        done = bool(done) or self.steps >= self.max_steps

        return state, inst_reward, done, {"rewards":rewards, "steps":self.steps}
