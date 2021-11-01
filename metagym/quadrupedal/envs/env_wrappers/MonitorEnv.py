import os
import gym
import numpy as np
import sys
import pybullet
from metagym.quadrupedal.robots import robot_config
from metagym import quadrupedal
from metagym.quadrupedal.robots import action_filter
from metagym.quadrupedal.envs.utilities.ETG_model import ETG_layer,ETG_model
from copy import copy

Param_Dict = {'torso':1.0,'up':0.3,'feet':0.2,'tau':0.1,'done':1,'velx':0,'badfoot':0.1,'footcontact':0.1}
Random_Param_Dict = {'random_dynamics':0,'random_force':0}
def EnvWrapper(env,param,sensor_mode,normal=0,ETG_T=0.5,enable_action_filter=False,
                reward_p=1,ETG=1,ETG_path="",ETG_T2=0.5,random_param=None,
                ETG_H=20,act_mode="traj",vel_d=0.6,vel_mode="max",
                task_mode="normal",step_y=0.05):
    env = ETGWrapper(env=env,ETG=ETG,ETG_T=ETG_T,ETG_path=ETG_path,
                    ETG_T2=ETG_T2,ETG_H=ETG_H,act_mode=act_mode,
                    task_mode=task_mode,step_y=step_y)
    env = ActionFilterWrapper(env=env,enable_action_filter=enable_action_filter)
    env = RandomWrapper(env=env,random_param=random_param)
    env = ObservationWrapper(env=env,ETG=ETG,sensor_mode=sensor_mode,normal=normal,ETG_H = ETG_H)
    env = RewardShaping(env=env,param=param,reward_p=reward_p,vel_d=vel_d,vel_mode=vel_mode)
    return env

class ActionFilterWrapper(gym.Wrapper):
    def __init__(self,env,enable_action_filter):
        gym.Wrapper.__init__(self, env) 
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.enable_action_filter = enable_action_filter and self.env.ETG.endswith("sac")
        if self.enable_action_filter:
            self._action_filter = self._BuildActionFilter()

    def reset(self,**kwargs):
        obs_all,info = self.env.reset(**kwargs)
        self._step_counter = 0
        if self.enable_action_filter:
            self._ResetActionFilter()
        return obs_all,info
    
    def step(self,action,**kwargs):
        if self.enable_action_filter:
            action = self._FilterAction(action)
        obs_all, rew, done, info = self.env.step(action)
        self._step_counter += 1
        return obs_all, rew, done, info


    def _BuildActionFilter(self):
        sampling_rate = 1 / self.env.env_time_step
        num_joints = 12
        a_filter = action_filter.ActionFilterButter(sampling_rate=sampling_rate,
                                                    num_joints=num_joints)
        return a_filter

    def _ResetActionFilter(self):
        self._action_filter.reset()

    def _FilterAction(self, action):
        # initialize the filter history, since resetting the filter will fill
        # the history with zeros and this can cause sudden movements at the start
        # of each episode
        if self._step_counter == 0:
            default_action = np.array([0,0,0]*4)
            self._action_filter.init_history(default_action)
            # for j in range(10):
            #     self._action_filter.filter(default_action)

        filtered_action = self._action_filter.filter(action)
        # print(filtered_action)
        return filtered_action    

class ObservationWrapper(gym.Wrapper):
    def __init__(self, env,ETG,sensor_mode,normal,ETG_H):
        gym.Wrapper.__init__(self, env) 
        # print("env_time:",self.env.env_time_step)
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.sensor_mode = sensor_mode
        self.normal = normal
        self.ETG_H = ETG_H
        self.ETG = ETG
        self.ETG_mean = np.array([2.1505982e-02,  3.6674485e-02, -6.0444288e-02,
                                2.4625482e-02,  1.5869144e-02, -3.2513142e-02,  2.1506395e-02,
                                3.1869926e-02, -6.0140789e-02,  2.4625063e-02,  1.1628972e-02,
                                -3.2163858e-02])
        self.ETG_std = np.array([4.5967497e-02,2.0340437e-01, 3.7410179e-01, 4.6187632e-02, 1.9441207e-01, 3.9488649e-01,
                                4.5966785e-02 ,2.0323379e-01, 3.7382501e-01, 4.6188373e-02 ,1.9457331e-01, 3.9302582e-01])
        if self.ETG:
            if "ETG" in self.sensor_mode.keys() and sensor_mode["ETG"] :
                sensor_shape = self.observation_space.high.shape[0]
                obs_h = np.array([1]*(sensor_shape+12))
                obs_l = np.array([0]*(sensor_shape+12))
                self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

            if "ETG_obs" in self.sensor_mode.keys() and sensor_mode["ETG_obs"] :
                sensor_shape = self.observation_space.high.shape[0]
                obs_h = np.array([1]*(sensor_shape+self.ETG_H))
                obs_l = np.array([0]*(sensor_shape+self.ETG_H))
                self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

        if "force_vec" in self.sensor_mode.keys() and sensor_mode["force_vec"]:
            sensor_shape = self.observation_space.high.shape[0]
            obs_h = np.array([1]*(sensor_shape+6))
            obs_l = np.array([0]*(sensor_shape+6))
            self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

        if "dynamic_vec" in self.sensor_mode.keys() and sensor_mode["dynamic_vec"]:
            sensor_shape = self.observation_space.high.shape[0]
            obs_h = np.array([1]*(sensor_shape+3))
            obs_l = np.array([0]*(sensor_shape+3))
            self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)
        
        if "yaw" in self.sensor_mode.keys() and sensor_mode["yaw"]:
            sensor_shape = self.observation_space.high.shape[0]
            obs_h = np.array([1]*(sensor_shape+2))
            obs_l = np.array([0]*(sensor_shape+2))
            self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)

        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0:
            self.time_steps = sensor_mode["RNN"]["time_steps"]
            self.time_interval = sensor_mode["RNN"]["time_interval"]
            self.sensor_shape = self.observation_space.high.shape[0]
            self.obs_history = np.zeros((self.time_steps*self.time_interval,self.sensor_shape))
            if sensor_mode["RNN"]["mode"] == "stack":
                obs_h = np.array([1]*(self.sensor_shape*(self.time_steps+1)))
                obs_l = np.array([0]*(self.sensor_shape*(self.time_steps+1)))
                self.observation_space = gym.spaces.Box(obs_l,obs_h,dtype=np.float32)
    
    def reset(self,**kwargs):
        obs,info = self.env.reset(**kwargs)
        self.dynamic_info = info["dynamics"]
        if self.ETG:
            if "ETG" in self.sensor_mode.keys() and self.sensor_mode["ETG"] :
                ETG_out = info["ETG_act"]
                if self.normal:
                    ETG_out = (ETG_out-self.ETG_mean)/self.ETG_std
                obs = np.concatenate((obs,ETG_out),axis = 0)

            if "ETG_obs" in self.sensor_mode.keys() and self.sensor_mode["ETG_obs"] :
                ETG_obs = info["ETG_obs"]
                obs = np.concatenate((obs,ETG_obs),axis = 0)

        if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
            force_vec = info["force_vec"]
            obs = np.concatenate((obs,force_vec),axis = 0)

        if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
            dynamic_vec = self.dynamic_info
            obs = np.concatenate((obs,dynamic_vec),axis = 0)

        if "yaw" in self.sensor_mode.keys() and self.sensor_mode["yaw"]:
            if "d_yaw" in kwargs.keys():
                d_yaw = kwargs["d_yaw"]
            else:
                d_yaw = 0
            yaw_now = info["pose"][-1]
            yaw_info = np.array([np.cos(d_yaw-yaw_now),np.sin(d_yaw-yaw_now)])
            obs = np.concatenate((obs,yaw_info),axis = 0)

        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0:
            self.obs_history = np.zeros((self.time_steps*self.time_interval,self.sensor_shape))
            obs_list = []
            for t in range(self.time_steps):
                obs_list.append(copy(self.obs_history[t*self.time_interval]))
            obs_list.append(copy(obs))
            self.obs_history[-1] = copy(obs)
            if self.sensor_mode["RNN"]["mode"]=="GRU":
                obs = np.stack(obs_list,axis=0)
            elif self.sensor_mode["RNN"]["mode"]=="stack":
                obs = np.array(obs_list).reshape(-1)

        return obs,info

    def step(self,action,**kwargs):
        obs, rew, done, info = self.env.step(action, **kwargs)
        if self.ETG:
            if "ETG" in self.sensor_mode.keys() and  self.sensor_mode["ETG"] :
                ETG_out = info["ETG_act"]
                if self.normal:
                    ETG_out = (ETG_out-self.ETG_mean)/self.ETG_std
                obs = np.concatenate((obs,ETG_out),axis = 0)

            if "ETG_obs" in self.sensor_mode.keys() and self.sensor_mode["ETG_obs"] :
                ETG_obs = info["ETG_obs"]
                obs = np.concatenate((obs,ETG_obs),axis = 0)
        
        if "force_vec" in self.sensor_mode.keys() and self.sensor_mode["force_vec"]:
            force_vec = info["force_vec"]
            obs = np.concatenate((obs,force_vec),axis = 0)

        if "dynamic_vec" in self.sensor_mode.keys() and self.sensor_mode["dynamic_vec"]:
            dynamic_vec = self.dynamic_info
            obs = np.concatenate((obs,dynamic_vec),axis = 0)

        if "yaw" in self.sensor_mode.keys() and self.sensor_mode["yaw"]:
            if "d_yaw" in kwargs.keys():
                d_yaw = kwargs["d_yaw"]
            else:
                d_yaw = 0
            yaw_now = info["pose"][-1]
            yaw_info = np.array([np.cos(d_yaw-yaw_now),np.sin(d_yaw-yaw_now)])
            obs = np.concatenate((obs,yaw_info),axis = 0)

        if "RNN" in self.sensor_mode.keys() and self.sensor_mode["RNN"]["time_steps"]>0:
            obs_list = []
            for t in range(self.time_steps):
                obs_list.append(copy(self.obs_history[t*self.time_interval]))
            obs_list.append(copy(obs))
            self.obs_history[:-1] = copy(self.obs_history[1:])
            self.obs_history[-1] = copy(obs)
            if self.sensor_mode["RNN"]["mode"]=="GRU":
                obs = np.stack(obs_list,axis=0)
            elif self.sensor_mode["RNN"]["mode"]=="stack":
                obs = np.array(obs_list).reshape(-1)
        return obs,rew,done,info

class ETGWrapper(gym.Wrapper):
    def __init__(self, env,ETG,ETG_T,ETG_path,ETG_T2,ETG_H=20,act_mode="traj",task_mode="normal",step_y=0.05):
        gym.Wrapper.__init__(self, env) 
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.ETG_T2 = ETG_T2
        self.ETG_T = ETG_T
        self.ETG_H = ETG_H
        self.act_mode = act_mode
        self.step_y = step_y
        self.task_mode = task_mode
        self.ETG = ETG
        phase = np.array([-np.pi/2,0])
        if self.ETG:
            self.ETG_agent = ETG_layer(self.ETG_T,self.env.env_time_step,self.ETG_H,0.04,phase,0.2,self.ETG_T2)
            self.ETG_weight = 1
            if len(ETG_path)>1 and os.path.exists(ETG_path):
                info = np.load(ETG_path)
                self.ETG_w = info["w"]
                self.ETG_b = info["b"]
            else:
                self.ETG_w = np.zeros((3,ETG_H))
                self.ETG_b = np.zeros(3)
            self.ETG_model = ETG_model(task_mode=self.task_mode,act_mode=act_mode,step_y=self.step_y)   
            self.last_ETG_act = np.zeros(12)
            self.last_ETG_obs = np.zeros(self.ETG_H)

    def reset(self,**kwargs):
        kwargs["info"] = True
        obs,info = self.env.reset(**kwargs)
        if self.ETG:
            if "ETG_w" in kwargs.keys() and kwargs["ETG_w"] is not None:
                self.ETG_w = kwargs["ETG_w"]
            if "ETG_b" in kwargs.keys() and kwargs["ETG_b"] is not None:
                self.ETG_b = kwargs["ETG_b"]
            self.ETG_agent.reset()
            state = self.ETG_agent.update2(t=self.env.get_time_since_reset())
            act_ref = self.ETG_model.forward(self.ETG_w,self.ETG_b,state)
            act_ref = self.ETG_model.act_clip(act_ref,self.robot)
            self.last_ETG_act = act_ref*self.ETG_weight
            info["ETG_obs"] = state[0]
            info["ETG_act"] = self.last_ETG_act
        return obs,info
    
    def step(self,action,**kwargs):
        if self.ETG:
            action = np.asarray(action).reshape(-1)+self.last_ETG_act
            state = self.ETG_agent.update2(t=self.env.get_time_since_reset())
            act_ref = self.ETG_model.forward(self.ETG_w,self.ETG_b,state)
            action_before = act_ref
            act_ref = self.ETG_model.act_clip(act_ref,self.robot)
            self.last_ETG_act = act_ref*self.ETG_weight
            obs, rew, done, info = self.env.step(action)
            info["ETG_obs"] = state[0]
            info["ETG_act"] = self.last_ETG_act
        else:
            obs, rew, done, info = self.env.step(action)
        return obs,rew,done,info
        
class RewardShaping(gym.Wrapper):
    def __init__(self, env,param,reward_p=1,vel_d=0.6,vel_mode="max"):
        gym.Wrapper.__init__(self, env)  
        self.param = param  
        self.reward_p = reward_p
        self.last_base10 = np.zeros((10,3))
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.vel_d = vel_d
        self.steps = 0
        self.vel_mode = vel_mode
        self.yaw_init = 0.0

    def reset(self,**kwargs):
        self.steps = 0
        obs,info = self.env.reset(**kwargs)
        self.yaw_init = info["yaw_init"]
        obs, rew, done, infos = self.env.step(np.zeros(self.action_space.high.shape[0]))
        self.last_basepose = info["base"]
        self.last_footposition = self.get_foot_world(info)
        base_pose = info["base"]
        self.last_base10 = np.tile(base_pose,(10,1))
        info["foot_position_world"] = copy(self.last_footposition)
        info["scene"] = "plane"
        if "d_yaw" in kwargs.keys():
            info['d_yaw'] = kwargs["d_yaw"]
        else:
            info['d_yaw'] = 0
        if self.render:
            self.line_id = self.draw_direction(info)
        return obs,info

    def step(self,action,**kwargs):
        self.steps+=1
        obs, rew, done, info = self.env.step(action, **kwargs)
        self.env_vec = np.array([0,0,0,0,0,0,0])
        posex = info["base"][0]
        for env_v in info["env_info"]:
            if posex+0.2 >= env_v[0] and posex+0.2 <= env_v[1]:
                self.env_vec = env_v[2]
                break
        if self.env_vec[0]:
            info["scene"] = "upslope"
        elif self.env_vec[1]:
            info["scene"] = "downslope"
        elif self.env_vec[2]:
            info["scene"] = "upstair"
        elif self.env_vec[3]:
            info["scene"] = "downstair"
        else:
            info["scene"] = "plane"
        v = (np.array(info["base"])-np.array(self.last_basepose))/0.026
        if "d_yaw" in kwargs.keys():
            info['d_yaw'] = kwargs["d_yaw"]
        else:
            info['d_yaw'] = 0
        donef = kwargs["donef"] if "donef" in kwargs.keys() else False
        info = self.reward_shaping(obs, rew, done, info,action,donef)
        info["vel"] = v
        rewards = 0
        done = self.terminate(info)
        if done:
            info["done"] = -1
        else:
            info["done"] = 0
        for key in Param_Dict.keys():
            if key in info.keys():
                # print(key)
                rewards+= info[key]
        info["velx"] = rew
        self.last_basepose = copy(info["base"])
        self.last_base10[1:,:] = self.last_base10[:9,:]
        self.last_base10[0,:] = np.array(info['base']).reshape(1,3)
        self.last_footposition = self.get_foot_world(info)
        info["foot_position_world"] = copy(self.last_footposition)
        if self.render:
            self.pybullet_client.removeUserDebugItem(self.line_id)
            self.line_id = self.draw_direction(info)
        return (obs, self.reward_p*rewards, done, info)


    def reward_shaping(self,obs, rew, done, info,action,donef,last_basepose=None,last_footposition=None):
        torso = self.re_torso(info,last_basepose=last_basepose)
        info['torso'] = self.param['torso']*torso
        if last_basepose is None:
            v = (np.array(info["base"])-np.array(self.last_basepose))/0.026
        else:
            v = (np.array(info["base"])-np.array(last_basepose))/0.026
        k = 1-self.c_prec(min(v[0],self.vel_d),self.vel_d,0.5)
        info['up'] = (self.param['up'])*self.re_up(info)*k
        info['feet'] = self.param['feet']*self.re_feet(info,last_footposition=last_footposition)
        info['tau'] = -self.param['tau']*info['energy']*k
        info['badfoot'] = -self.param['badfoot']*self.robot.GetBadFootContacts()
        lose_contact_num = np.sum(1.0-np.array(info["real_contact"]))
        info['footcontact'] = -self.param['footcontact']*max(lose_contact_num-2,0)
        return info
    
    def draw_direction(self,info):
        pose = info['base']
        if self.render:
            id = self.pybullet_client.addUserDebugLine(lineFromXYZ=[pose[0],pose[1],0.6],
                                                    lineToXYZ=[pose[0]+np.cos(info['d_yaw']),pose[1]+np.sin(info['d_yaw']),0.6],
                                                    lineColorRGB=[1,0,1],lineWidth=2)
        return id

    def terminate(self,info):
        rot_mat = info["rot_mat"]
        pose = info["pose"]
        footposition = copy(info["footposition"])
        footz = footposition[:,-1]
        base = info["base"]
        base_std = np.sum(np.std(self.last_base10,axis=0))
        return rot_mat[-1]<0.5  or np.mean(footz)>-0.1 or np.max(footz)>0  or (base_std<=2e-4 and self.steps>=10) or abs(pose[-1])>0.6

    def _calc_torque_reward(self):
        energy = self.robot.GetEnergyConsumptionPerControlStep()
        return -energy

    def re_still(self,info):
        v = (np.array(info["base"])-np.array(self.last_basepose))/0.026
        return -np.linalg.norm(v)
    
    def re_standupright(self,info):
        still = self.re_still(info)
        up = self.re_up(info)
        return self.re_rot(info,still+up)

    def re_up(self,info):
        posex = info["base"][0]
        env_vec = np.zeros(7)
        for env_v in info["env_info"]:
            if posex+0.2 >= env_v[0] and posex+0.2 <= env_v[1]:
                env_vec = env_v[2]
                break
        pose = copy(info["pose"])
        roll = pose[0]
        pitch = pose[1]
        if env_vec[0]:
            pitch += abs(env_vec[4])
        elif env_vec[1]:
            pitch -= abs(env_vec[4])
        r = np.sqrt(roll**2+pitch**2)
        return 1-self.c_prec(r,0,0.4)

    def re_rot(self,info,r):
        pose = copy(info["pose"])
        yaw = pose[-1]
        k1 = 1-self.c_prec(yaw,info['d_yaw'],0.5)
        k2 = 1-self.c_prec(yaw,info['d_yaw']+2*np.pi,0.5)
        k3 = 1-self.c_prec(yaw,info['d_yaw']-2*np.pi,0.5)
        k = max(k1,k2,k3)
        return min(k*r,r)

    def c_prec(self,v,t,m):
        if m<1e-5:
            print(m)
        w = np.arctanh(np.sqrt(0.95))/m
        return np.tanh(np.power((v-t)*w,2))

    def re_feet(self,info,vd=[1,0,0],last_footposition=None):
        vd[0] = np.cos(info['d_yaw'])
        vd[1] = np.sin(info['d_yaw'])
        posex = info["base"][0]
        env_vec = np.zeros(7)
        for env_v in info["env_info"]:
            if posex+0.2 >= env_v[0] and posex+0.2 <= env_v[1]:
                env_vec = env_v[2]
                break
        if env_vec[0]:
            vd[0] *= abs(np.cos(env_vec[4]))
            vd[1] *= abs(np.cos(env_vec[4]))
            vd[2] = abs(np.sin(env_vec[4]))
        elif env_vec[1]:
            vd[0] *= abs(np.cos(env_vec[4]))
            vd[1] *= abs(np.cos(env_vec[4]))
            vd[2] = -abs(np.sin(env_vec[4]))
        foot = self.get_foot_world(info)
        if last_footposition is None:
            d_foot = (foot-self.last_footposition)/0.026
        else:
            d_foot = (foot-last_footposition)/0.026
        v_sum = 0
        contact = copy(info["real_contact"])
        for i in range(4):
            v = d_foot[i]
            v_ = v[0]*vd[0]+v[1]*vd[1]+v[2]*vd[2]
            r = min(v_,self.vel_d)/4.0
            v_sum += min(r,1.0*r)
        return self.re_rot(info,v_sum)

    def get_foot_world(self,info={}):
        if "footposition" in info.keys():
            foot = np.array(info["footposition"]).transpose()
            rot_mat = np.array(info["rot_mat"]).reshape(-1,3)
            base = np.array(info["base"]).reshape(-1,1)
        else:
            foot = np.array(self.robot.GetFootPositionsInBaseFrame()).transpose()
            rot_quat = self.robot.GetBaseOrientation()
            rot_mat = np.array(self.pybullet_client.getMatrixFromQuaternion(rot_quat)).reshape(-1,3)
            base = np.array(self.robot.GetBasePosition()).reshape(-1,1)
            print("no!")
        foot_world = rot_mat.dot(foot)+base
        return foot_world.transpose()

    def re_torso(self,info,vd = [1,0,0],last_basepose = None):
        if last_basepose is None:
            v = (np.array(info["base"])-np.array(self.last_basepose))/0.026
        else:
            v = (np.array(info["base"])-np.array(last_basepose))/0.026
        vd[0] = np.cos(info['d_yaw'])
        vd[1] = np.sin(info['d_yaw'])
        posex = info["base"][0]
        env_vec = np.zeros(7)
        for env_v in info["env_info"]:
            if posex+0.2 >= env_v[0] and posex+0.2 <= env_v[1]:
                env_vec = env_v[2]
                break
        if env_vec[0]:
            vd[0] *= abs(np.cos(env_vec[4]))
            vd[1] *= abs(np.cos(env_vec[4]))
            vd[2] = abs(np.sin(env_vec[4]))
        elif env_vec[1]:
            vd[0] *= abs(np.cos(env_vec[4]))
            vd[1] *= abs(np.cos(env_vec[4]))
            vd[2] = -abs(np.sin(env_vec[4]))
        if self.vel_mode == "max":
            v_ = v[0]*vd[0]+v[1]*vd[1]+v[2]*vd[2]
            v_reward = min(self.vel_d,v_)
        elif self.vel_mode == "equal":
            v_ = v[0]*vd[0]+v[1]*vd[1]+v[2]*vd[2]
            v_diff = abs(v_-self.vel_d)
            v_reward = np.exp(-5*v_diff)
        return self.re_rot(info,v_reward)

class RandomWrapper(gym.Wrapper):
    def __init__(self, env,random_param):
        gym.Wrapper.__init__(self, env)  
        self.random_param = random_param if random_param is not None else {} 
        self.robot = self.env.robot
        self.pybullet_client = self.env.pybullet_client
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render = self.env.rendering_enabled
    
    def generate_randomforce(self):
        force_position = (np.random.random(3)-0.5)*2*np.array([0.2,0.05,0.05])
        force_vec = np.random.uniform(low=-1,high=1,size=(3,))*np.array([0.5,1,0.05])
        force_vec = force_vec/np.linalg.norm(force_vec)*np.random.uniform(20,50)
        return force_position,force_vec

    def draw_forceline(self,force_position,force_vec):
        if self.render:
            self.pybullet_client.addUserDebugLine(lineFromXYZ=force_position,lineToXYZ=force_position+force_vec/50,
                                                    parentObjectUniqueId=self.robot.quadruped,
                                                    parentLinkIndex=-1,lineColorRGB=[1,0,0])
    def random_dynamics(self,info):
        footfriction = 1
        footfriction_normal = 0

        basemass = self.robot.GetBaseMassesFromURDF()[0]
        basemass_ratio_normal = 0

        baseinertia = self.robot.GetBaseInertiasFromURDF()
        baseinertia_ratio_normal = np.zeros(3)

        legmass = self.robot.GetLegMassesFromURDF()
        legmass_ratio_normal = np.zeros(3)

        leginertia = self.robot.GetLegInertiasFromURDF()
        leginertia_ratio_normal = np.zeros(3)

        control_latency = 0
        control_latency_normal = -1

        joint_friction = 0.025
        joint_friction_normal = [0]
        joint_friction_vec = np.array([joint_friction]*12)

        spin_friction = 0.2
        spin_friction_normal = 0
            
            
        if "random_dynamics" in self.random_param.keys() and self.random_param["random_dynamics"]:
            #friction
            # footfriction = np.random.uniform(1,2.5)
            # footfriction_normal = footfriction-1

            #basemass
            # basemass_ratio = np.random.uniform(0.8,1.2)
            # basemass_ratio_normal = (basemass_ratio-1)/0.2
            # basemass = basemass*basemass_ratio

            #baseinertia
            # baseinertia_ratio = np.random.uniform(0.8,1.2,3)
            # baseinertia_ratio_normal = (baseinertia_ratio-1)/0.2
            # baseinertia = baseinertia[0]
            # baseinertia = [(baseinertia[0]*baseinertia_ratio[0],baseinertia[1]*baseinertia_ratio[1],baseinertia[2]*baseinertia_ratio[2])]
            
            #legmass
            # legmass_ratio = np.random.uniform(0.8,1.2,3)
            # legmass_ratio_normal = (legmass_ratio-1)/0.2
            # legmass = legmass*np.array([legmass_ratio[0],legmass_ratio[1],legmass_ratio[2]]*4)

            #leginertia
            # leginertia_ratio = np.random.uniform(0.8,1.2,3)
            # leginertia_ratio_normal = (leginertia_ratio-1)/0.2
            # leginertia_new = []
            # for lg in leginertia:
            #      leginertia_new.append(leginertia_ratio*lg)
            # leginertia = copy(leginertia_new)

            # #control_latency
            control_latency = np.random.uniform(0.01,0.02)
            control_latency_normal = (control_latency-0.01)/0.01
            print("latency:",control_latency)
            #joint_friction
            # joint_friction = np.random.random(1)*0.05
            # joint_friction_normal = (joint_friction/0.05-0.5)*2
            # joint_friction_vec = np.array([joint_friction]*12)

            #spin_friction
            # spin_friction = np.random.uniform(0,0.4)
            # spin_friction_normal = (spin_friction-0.2)*5

        dynamics_vec = np.concatenate(([footfriction_normal],[basemass_ratio_normal],
                                        baseinertia_ratio_normal,legmass_ratio_normal,
                                        leginertia_ratio_normal,[control_latency_normal],
                                        joint_friction_normal,[spin_friction_normal]),axis=0)
        self.robot.SetFootFriction(footfriction)
        self.robot.SetBaseMasses([basemass])
        self.robot.SetBaseInertias(baseinertia)
        self.robot.SetLegMasses(legmass)
        self.robot.SetLegInertias(leginertia)
        self.robot.SetControlLatency(control_latency)
        self.robot.SetJointFriction(joint_friction_vec)
        self.robot.SetFootSpinFriction(spin_friction)
        info['dynamics'] = dynamics_vec
        return info


    def reset(self,**kwargs):
        # infos = self.random_dynamics({})
        obs,info = self.env.reset(**kwargs)
        info['dynamics'] = np.array([info['latency'],info["footfriction"],info['basemass']])
        force_info = np.zeros(6)
        if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
            self.pybullet_client.removeAllUserDebugItems()
            self.force_position, self.force_vec = self.generate_randomforce()
            self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                            posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
            self.draw_forceline(self.force_position,self.force_vec)
            force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
        # info = self.random_dynamics(info)
        info['force_vec'] = force_info
        # print("init_info",info)
        return obs,info
    
    def step(self,action,**kwargs):
        force_info = np.zeros(6)
        obs, rew, done, info = self.env.step(action, **kwargs)
        if "random_force" in self.random_param.keys() and self.random_param["random_force"]:
            # New random force
            if self.env.env_step_counter % 100 == 0:
                self.force_position, self.force_vec = self.generate_randomforce()
                self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                    posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
                self.draw_forceline(self.force_position,self.force_vec)
                force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
            # Apply force
            elif self.env.env_step_counter % 100 <50:
                self.pybullet_client.applyExternalForce(objectUniqueId=self.robot.quadruped,linkIndex=-1,forceObj=self.force_vec,
                                    posObj=self.force_position,flags=self.pybullet_client.LINK_FRAME)
                self.draw_forceline(self.force_position,self.force_vec)
                force_info = np.concatenate((self.force_position/np.array([0.2,0.05,0.05]),self.force_vec/50),axis=0)
            # delete line
            elif self.env.env_step_counter % 100 ==50:
                self.pybullet_client.removeAllUserDebugItems() 
        info['force_vec'] = force_info 
        return obs, rew, done, info

