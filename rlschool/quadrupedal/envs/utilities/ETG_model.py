import numpy as np
from copy import copy
Param_Dict = {'stand':0,'torso':0,'up':0,'feet':0,'contact':0,'walk':0,'footh':0,'instab':0,'col':0,'slip':0,'velx':0,'foot':0,'tau':0,'vely':0,'height':0,'footsym':0,'footpos':0,'act':0,'joint':0,'gaitref':0,'done':1}
ACTION_DIM = 12
base_foot = np.array([ 0.18,-0.15,-0.23,0.18,0.148,-0.23,\
                        -0.18,-0.14,-0.23,-0.18,0.135,-0.23])

class ETG_layer():
    def __init__(self,T,dt,H,sigma_sq,phase,amp,T2_radio):
        #T2_ratio mean the ratio forward t
        self.dt = dt
        self.T = T
        self.t = 0
        self.H = H
        self.sigma_sq = sigma_sq
        self.phase = phase
        self.amp = amp
        self.u = []
        self.omega = 2.0*np.pi/T
        self.T2_ratio = T2_radio
        for h in range(H):
            t_now = h*self.T/(H-0.9)
            self.u.append(self.forward(t_now))
        self.u = np.asarray(self.u).reshape(-1,2)
        self.TD = 0
        
    def forward(self,t):
        x = []
        for i in range(self.phase.shape[0]):
            x.append(self.amp*np.sin(self.phase[i]+t*self.omega))
        return np.asarray(x).reshape(-1)
    
    def update(self,t=None):
        time = t if t is not None else self.t
        x = self.forward(time)
        self.t += self.dt
        r = []
        for i in range(self.H):
            dist = np.sum(np.power(x-self.u[i],2))/self.sigma_sq
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)
        return r

    def update2(self,t=None,info=None):
        time = t if t is not None else self.t
        x = self.forward(time)
        x2 = self.forward(time+self.T2_ratio*self.T)
        self.t += self.dt
        r = []
        for i in range(self.H):
            dist = np.sum(np.power(x-self.u[i],2))/self.sigma_sq
            r.append(np.exp(-dist))
        r = np.asarray(r).reshape(-1)
        r2 = []
        for i in range(self.H):
            dist = np.sum(np.power(x2-self.u[i],2))/self.sigma_sq
            r2.append(np.exp(-dist))
        r2 = np.asarray(r2).reshape(-1)
        return (r,r2)
    def observation_T(self):
        ts = np.arange(0,self.T,self.dt)
        x = {t:self.forward(t) for t in ts}
        r_all = {}
        for j in ts:
            r = []
            for i in range(self.H):
                dist = np.sum(np.power(x[j]-self.u[i],2))/self.sigma_sq
                r.append(np.exp(-dist))
            r_all[j] = np.asarray(r).reshape(-1)
        return r_all

    def get_phase(self):
        return self.forward(self.t-self.dt)

    def reset(self):
        self.t = 0
        self.TD = 0

class ETG_model():
    def __init__(self,task_mode="normal",act_mode="traj",step_y=0.5):
        self.act_mode = act_mode
        self.pose_ori = np.array([0,0.9,-1.8]*4)
        self.task_mode = task_mode
        if self.task_mode == "cave":
            self.base_foot = np.array([ 0.18+0.02,-0.15,-0.1,0.18+0.02,0.148,-0.1,\
                        -0.18+0.02,-0.14,-0.1,-0.18+0.02,0.135,-0.1])
        else:
            self.base_foot = np.array([ 0.18,-0.15,-0.23,0.18,0.148,-0.23,\
                        -0.18,-0.14,-0.23,-0.18,0.135,-0.23])
        if self.task_mode == "balance":
            self.base_foot[1] = -step_y
            self.base_foot[4] = step_y
            self.base_foot[7] = -step_y
            self.base_foot[10] = step_y
    def forward(self,w,b,x):
        x1 = np.asarray(x[0]).reshape(-1,1)
        x2 = np.asarray(x[1]).reshape(-1,1)
        act1 = w.dot(x1).reshape(-1)+b
        act2 = w.dot(x2).reshape(-1)+b
        new_act = np.zeros(ACTION_DIM)
        if self.task_mode == "gallop":
            new_act[:3] = act1.copy()
            new_act[3:6] = act1.copy()
            new_act[6:9] = act2.copy()
            new_act[9:] = act2.copy()
        else:
            new_act[:3] = act1.copy()
            new_act[3:6] = act2.copy()
            new_act[6:9] = act2.copy()
            new_act[9:] = act1.copy()
        return new_act
    def act_clip(self,new_act,env):
        if self.act_mode == "pose":
            #joint control mode
            act = np.tanh(new_act)*np.array([0.1,0.7,0.7]*4)
        elif self.act_mode == "traj":
            #foot trajectory mode
            act = np.zeros(12)
            for i in range(4):
                delta = new_act[i*3:(i+1)*3].copy()
                while(1):
                    index,angle = env.ComputeMotorAnglesFromFootLocalPosition(i,delta+base_foot[i*3:(i+1)*3])
                    if np.sum(np.isnan(angle))==0:
                        break
                    delta *= 0.95
                act[index] = np.array(angle)
            act -= self.pose_ori
        return act
