import pybullet as p
import numpy as np
from metagym.quadrupedal.envs.utilities.pose3d import QuaternionFromAxisAngle
import time

STEP_HEIGHT_INTERVAL = 0.002
SLOPE_INTERVAL = 0.02
STEP_WIDTH_INTERVAL = 0.02
STEP_HEIGHT = np.arange(0.08,0.101,STEP_HEIGHT_INTERVAL)
SLOPE = np.arange(0.3,0.501,SLOPE_INTERVAL)
STEP_WIDTH = np.arange(0.26,0.401,STEP_WIDTH_INTERVAL)
STEP_PER_NUM = 5
DELTA_X = 1
FRICTION = 5.0
def upstair_terrain(stepwidth=0.33,stepheight=0.05,slope=0.05,stepnum=40,mode="terrain-fix",env_vecs=[]):
    add_height = 0
    # print("mode:",mode)
    env_info = []
    if mode == "stair-fix":
        boxHalfLength = stepwidth
        boxHalfWidth = 2.5
        boxHalfHeight = stepheight
        sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
        sth=stepheight
        steplen = stepwidth
        basez = - stepheight
        for i in range(stepnum):
            id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [0.66+i*steplen,0,basez+(i+1)*sth],baseOrientation=[0.0,0.0,0.0,1])
            p.changeDynamics(id,-1,lateralFriction=FRICTION)
        env_info.append([-10,100,np.array([0,0,1,0,0,stepheight,stepwidth])])
    elif mode == "stair-var":
        basez = 0
        for i in range(5):
            boxHalfLength = stepwidth
            boxHalfWidth = 2.5
            boxHalfHeight = stepheight+i*0.01
            sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
            sth=boxHalfHeight
            steplen = stepwidth
            stepnums = 8
            for j in range(stepnums):
                id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [0.66+(i*stepnums+j)*steplen,0,basez+j*sth],baseOrientation=[0.0,0.0,0.0,1])
                p.changeDynamics(id,-1,lateralFriction=1.0)
            basez = basez + stepnums*sth
    elif mode == "downstair":
        downstair_terrain(stepwidth,stepheight)
        add_height = stepheight*stepnum
        env_info.append([-10,100,np.array([0,0,0,1,0,stepheight,stepwidth])])
    elif mode == "slope":
        upslope_terrain(slope)
        if slope<0:
            add_height = 100*np.sin(abs(slope))-np.tan(abs(slope))+abs(slope)*0.15
            env_info.append([-10,100,np.array([0,1,0,0,slope,0,0])])
        else:
            env_info.append([-10,100,np.array([1,0,0,0,slope,0,0])])
    elif mode.endswith("random") or mode == "special":
        if mode == "special":
            env_vectors = env_vecs
        else:
            env_vectors = generate_env_vec(mode,10)
        # env_vectors = [np.array([0.  , 0.  , 0.  , 1.  , 0.08, 0.07, 0.3 ]), np.array([0.  , 0.  , 1.  , 0.  , 0.12, 0.05, 0.28])]
        deltaz = cal_basez(env_vectors,STEP_PER_NUM,DELTA_X)
        basex = -1
        basez = 0
        # print(env_vectors)
        # print("deltaz:",deltaz)
        if deltaz < 0:
            add_height = abs(deltaz)
            basez = abs(deltaz)
        # print("init:",basex,basez)
        last_x = basex
        basex,basez = subplane(basex,basez,0.5)
        env_info.append([last_x,basex,np.array([0,0,0,0,0,0,0])])
        for i in range(len(env_vectors)):
            env_vec = env_vectors[i]
            last_x = basex
            if env_vec[0]:
                env_vec[5] = 0
                env_vec[6] = 0
                basex,basez = subslope(basex=basex,basez=basez,slope=env_vec[4],endx = basex+DELTA_X)
            elif env_vec[1]:
                env_vec[5] = 0
                env_vec[6] = 0
                basex,basez = subslope(basex=basex,basez=basez,slope=-env_vec[4],endx = basex+DELTA_X)
            elif env_vec[2]:
                env_vec[4] = 0
                basex,basez = substair(basex=basex,basez=basez,stepwidth=env_vec[6],stepheight=env_vec[5],stepnum=STEP_PER_NUM,sign="up")
            elif env_vec[3]:
                env_vec[4] = 0
                basex,basez = substair(basex=basex,basez=basez,stepwidth=env_vec[6],stepheight=env_vec[5],stepnum=STEP_PER_NUM,sign="down")
            else:
                env_vec = np.zeros(7)
                basex,basez = subplane(basex,basez,basex+1)
            env_info.append([last_x,basex,env_vec])
            last_x = basex
            # print("{}: basex{} basez{}".format(i,basex,basez))
            if i < len(env_vectors)-1:
                next_env_vec = env_vectors[i+1]
                if env_vec[3] and (next_env_vec[0] or next_env_vec[2]):
                    basex,basez = subplane(basex,basez,basex+0.5)
                elif env_vec[0] and (next_env_vec[1] or next_env_vec[2]):
                    basex,basez = subplane(basex,basez,basex+0.5)
                elif env_vec[1] and (next_env_vec[2] or next_env_vec[0]):
                    basex,basez = subplane(basex,basez,basex+0.5)
                elif env_vec[2] and (next_env_vec[0] or next_env_vec[1]):
                    basex,basez = subplane(basex,basez,basex+0.2)
            if last_x != basex:
                env_info.append([last_x,basex,np.array([0,0,0,0,0,0,0])])
                last_x = basex
            # print("{}: basex{} basez{}".format(i,basex,basez))
        if basez >0:
            deltax = basez/np.tan(0.4)
            basex,basez = subslope(basex=basex,basez=basez,slope=-0.4,endx = basex+deltax)
            env_info.append([last_x,basex,np.array([0,1,0,0,-0.4,0,0])])
        # print(env_info)
    elif mode == "balance_beam":
        add_height = 5
        balance_beam(stepwidth,stepheight)
        env_info.append([-10,100,np.zeros(7)])
    elif mode == "gallop":
        add_height = 5
        gallop(stepwidth)
        env_info.append([-10,100,np.zeros(7)])
    elif mode == "hurdle":
        hurdle(stepwidth,stepheight)
        env_info.append([-10,100,np.zeros(7)])
    elif mode == "cave":
        cave(stepwidth,stepheight)
        env_info.append([-10,100,np.zeros(7)])
    return add_height,env_info
    # elif mode == "random":

def cal_basez(env_vectors,stepnum,deltax):
    endz = 0
    min_z = 0
    for env_vec in env_vectors:
        if env_vec[0]:
            endz += np.tan(env_vec[4])*deltax
        elif env_vec[1]:
            endz -= np.tan(env_vec[4])*deltax
        elif env_vec[2]:
            endz += stepnum*env_vec[5]
        elif env_vec[3]:
            endz -= stepnum*env_vec[5]
        if min_z > endz:
            min_z = endz
    return min_z

def generate_env_vec(mode,num):
    if mode.startswith("upslope"):
        env_heads = np.array([0]*num)
    elif mode.startswith("downslope"):
        env_heads = np.array([1]*num)
    elif mode.startswith("upstair"):
        env_heads = np.array([2]*num)
    elif mode.startswith("downstair"):
        env_heads = np.array([3]*num)
    else:
        env_heads = np.random.choice(4,num)
    stepheights = np.random.choice(STEP_HEIGHT,num)
    stepwidths = np.random.choice(STEP_WIDTH,num)
    slopes = np.random.choice(SLOPE,num)
    # print("stepheights:",stepheights)
    env_vectors = []
    for PN in range(num):
        env_vec = np.zeros(7)
        env_vec[env_heads[PN]] = 1
        env_vec[4] = slopes[PN]
        env_vec[5] = stepheights[PN]
        env_vec[6] = stepwidths[PN]
        env_vectors.append(env_vec)
    return env_vectors


def subplane(basex=0,basez=0,endx=0):
    boxHalfLength = (endx-basex)/2.0
    boxHalfWidth = 2.5
    boxHalfHeight = 0.01
    if basez <= 0.01:
        basez = 0.01
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [basex+boxHalfLength,0,basez-boxHalfHeight],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id,-1,lateralFriction=FRICTION)
    return(endx,basez)


def substair(basex=0,basez=0,stepwidth=0.33,stepheight=0.05,stepnum=40,sign="up"):
    boxHalfLength = stepwidth/2
    boxHalfWidth = 2.5
    boxHalfHeight = stepheight/2
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    if sign == "up":
        for i in range(stepnum):
            id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [basex+(i+0.5)*stepwidth,0,basez+(i+0.5)*stepheight],baseOrientation=[0.0,0.0,0.0,1])
            p.changeDynamics(id,-1,lateralFriction=FRICTION)
        return (basex+stepnum*stepwidth,basez+stepnum*stepheight)
    else:
        for i in range(stepnum):
            id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                                basePosition = [basex+(i+0.5)*stepwidth,0,basez-(i+0.5)*stepheight],baseOrientation=[0.0,0.0,0.0,1])
            p.changeDynamics(id,-1,lateralFriction=FRICTION)
        return (basex+stepnum*stepwidth,basez-stepnum*stepheight)        

def subslope(basex=0,basez=0,slope=0.1,endx = 0):
    boxHalfWidth = 2.5
    boxHalfHeight = 0.01
    boxHalfLength = abs((endx-basex)/np.cos(slope))/2.0
    # quaterion = QuaternionFromAxisAngle(axis=[0,1,0],angle=-slope)
    quaterion = [0,np.sin(-slope/2.0),0,np.cos(-slope/2.0)]
    # print("constract:",quaterion,q_new)
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                        basePosition = [basex+boxHalfLength*np.cos(slope),0,basez+boxHalfLength*np.sin(slope)],baseOrientation=quaterion)
    p.changeDynamics(id,-1,lateralFriction=FRICTION)
    return (endx,basez+np.sin(slope)*boxHalfLength*2)
    


def downstair_terrain(stepwidth,stepheight,stepnum=40):
    boxHalfLength = 2.5
    boxHalfWidth = 2.5
    boxHalfHeight = stepheight
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    #mass = 1
    #block=p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
    #                        basePosition = [-2,0,-0.1],baseOrientation=[0.0,0.1,0.0,1])
    sth=stepheight
    steplen = stepwidth
    basez = stepheight*stepnum
    for i in range(stepnum):
        id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [0.5-boxHalfLength+i*steplen,0,basez-(i+1)*sth],baseOrientation=[0.0,0.0,0.0,1])
        p.changeDynamics(id,-1,lateralFriction=FRICTION)

def upslope_terrain(slope=0.05):
    boxHalfLength = 50
    boxHalfWidth = 2.5
    boxHalfHeight = 0.01
    quaterion = QuaternionFromAxisAngle(axis=[0,1,0],angle=-slope)
    # print(quaterion)
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    if slope>0:
        id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [0.15+boxHalfLength*np.cos(slope),0,boxHalfLength*np.sin(slope)],baseOrientation=quaterion)
    else:
        id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [-1+boxHalfLength*np.cos(slope),0,-boxHalfLength*np.sin(slope)],baseOrientation=quaterion)
    p.changeDynamics(id,-1,lateralFriction=FRICTION)

def balance_beam(stepwidth=0.1,stepheight=1):
    boxHalfLength = 2.5
    boxHalfWidth = 2.5
    boxHalfHeight = 0.01
    sh_colBox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    id_front = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [-2.3,0,5],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_front,-1,lateralFriction=FRICTION)
    id_end = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,
                            basePosition = [stepheight+2.7,0,5],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_front,-1,lateralFriction=FRICTION)
    balance_beam = p.createCollisionShape(p.GEOM_BOX,halfExtents=[stepheight/2.0,stepwidth,boxHalfHeight])
    id_balance = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = balance_beam,
                            basePosition = [stepheight/2.0+0.2,0,5],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_balance,-1,lateralFriction=FRICTION)

def gallop(stepwidth = 0.3,stepnum=30):
    boxHalfLength = 2.5 
    boxHalfWidth = 2.5
    boxHalfHeight = 0.01
    land_Box = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    id_front = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = land_Box,
                            basePosition = [-2.2,0,5],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_front,-1,lateralFriction=FRICTION)
    cliff_len = 0.5
    cliff_Box = p.createCollisionShape(p.GEOM_BOX,halfExtents=[cliff_len/2.0,boxHalfWidth,boxHalfHeight])
    current_x = 0.3
    for i in range(stepnum):
        id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = cliff_Box,
                            basePosition = [current_x+stepwidth+cliff_len/2.0,0,5],baseOrientation=[0.0,0.0,0.0,1])
        p.changeDynamics(id,-1,lateralFriction=FRICTION)
        current_x += stepwidth + cliff_len
    id_back = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = land_Box,
                            basePosition = [current_x+boxHalfWidth/2.0,0,5],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_back,-1,lateralFriction=FRICTION)

def hurdle(stepwidth = 0.3,stepheight=0.2, stepnum = 30):
    boxHalfLength = 0.01
    boxHalfWidth = 4
    boxHalfHeight = stepheight/2.0
    current_x = stepwidth/2.0
    land_Box = p.createCollisionShape(p.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
    for i in range(stepnum):
        id = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = land_Box,
                            basePosition = [current_x,0,stepheight/2.0],baseOrientation=[0.0,0.0,0.0,1])
        p.changeDynamics(id,-1,lateralFriction=FRICTION)
        current_x += stepwidth

def cave(stepwidth = 0.25,stepheight = 0.4):
    upbox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[10,stepwidth,0.01])
    id_up = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = upbox,
                            basePosition = [10+0.3,0,stepheight],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_up,-1,lateralFriction=FRICTION)

    leftbox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[15,0.01,stepheight/2.0])
    id_left = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = leftbox,
                            basePosition = [10,stepwidth,stepheight/2.0],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_left,-1,lateralFriction=FRICTION)

    rightbox = p.createCollisionShape(p.GEOM_BOX,halfExtents=[15,0.01,stepheight/4.0])
    id_right = p.createMultiBody(baseMass=0,baseCollisionShapeIndex = rightbox,
                            basePosition = [10,-stepwidth,stepheight/4.0],baseOrientation=[0.0,0.0,0.0,1])
    p.changeDynamics(id_right,-1,lateralFriction=FRICTION)
