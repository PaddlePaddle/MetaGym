# 四轴飞行器环境

如果该环境对您的研究有帮助，请考虑引用:

```txt
@misc{Quadrotor,
    author = {Yang Xue, Fan Wang and Bo Zhou},
    title = {{A configurable lightweight simulator of quad-rotor helicopter}},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/PaddlePaddle/RLSchool/tree/master/rlschool/quadrotor}},
}
```

## 安装

```
pip install rlschool
```

## 测试环境可视化功能

```sh
# 避障任务, 'no_collision'
python -m rlschool.quadrotor.env

# 速度控制任务, 'velocity_control'
python -m rlschool.quadrotor.env velocity_control

# 悬浮控制任务, 'hovering_control'
python -m rlschool.quadrotor.env hovering_control
```

环境正确安装后，飞行器飞行渲染效果如下：

* 避障任务

<img src="demo/demo_no_collision.gif" width="400"/>

* 速度控制任务

<img src="demo/demo_velocity_control.gif" width="400"/>

* 悬浮控制任务

<img src="demo/demo_hovering_control.gif" width="400"/>

## 环境创建及使用

四轴飞行器环境遵循标准的[Gym][gym] APIs接口来创建、运行和渲染环境。目前，四轴飞行器支持3种任务：避障任务、速度控制任务、悬浮控制任务。任务由创建环境时的`task`参数设定。

创建避障任务的示例代码如下：

```python
from rlschool import make_env
env = make_env("Quadrotor", task="no_collision", map_file=None, simulator_conf=None)
env.reset()
```

当参数`map_file`为默认值`None`时，模拟器世界是将使用100x100的平底作为地图。一旦飞行器落下，即认为击中障碍物，环境终止。地图文件是如[default_map.txt][map_example]格式的文本文件，其中每个数字表示对应位置障碍墙的高度。`-1`标记了飞行器的初识位置。需要时，可以设定`map_file`为用户自己生成的地图配置文件。

当参数`simulator_conf`是默认值`None`时，将会使用默认的模拟器配置[config.json][default_sim_conf]。如果用户需要自主设定四轴飞行器的动力学参数，可以将`simulator_conf`设定为新的_config.json_的路径。

创建速度控制任务的示例代码如下：

```python
from rlschool import make_env
env = make_env("Quadrotor", task="velocity_control", seed=0)
env.reset()
```

其中，`seed`参数是用来采样目标速度序列的随机种子，不同随机种子将生成不同的速度控制任务。

创建悬浮控制任务的示例代码如下：

```python
from rlschool import make_env
env = make_env("Quadrotor", task="hovering_control")
env.reset()
```

## 动作空间

四轴飞行器的动作代表施加在4个螺旋桨发动机的4个电压值，安装默认配置文件[config.json](default_sim_conf)，电压的范围在`[0.10, 15.0]`。

用户可以通过一些特殊的动作理解控制电压如何操作四轴飞行器，例如，动作`[1.0, 1.0, 1.0, 1.0]`将使无初速度的飞行器垂直向上或向下运动。

```python
from rlschool import make_env

env = make_env("Quadrotor", task="no_collision")
env.reset()
env.render()

reset = False
while not reset:
    state, reward, reset, info = env.step([1.0, 1.0, 1.0, 1.0])
    env.render()
```

## 状态

四轴飞行器的状态可以分为3类：传感器测量数据，飞行状态和任务相关状态。可以通过`env.step`返回的第四个量`info`获得Python字典形式的状态值。

### 传感器测量数据

* `acc_x`: 加速度计在x轴方向上的测量值。
* `acc_y`: 加速度计在y轴方向上的测量值。
* `acc_z`: 加速度计在z轴方向上的测量值。
* `gyro_x`: 陀螺仪在x轴方向上的测量值。
* `gyro_y`: 陀螺仪在y轴方向上的测量值。
* `gyro_z`: 陀螺仪在z轴方向上的测量值。
* `z`: 气压计测量的高度数据，为方便认为是到地面的距离。
* `pitch`: 飞行器绕y轴的逆时针转动角度。
* `roll`: 飞行器绕x轴的逆时针转动角度。
* `yaw`: 飞行器绕z轴的逆时针转动角度。

这里需要说明的是，`pitch`, `roll`, `yaw`是陀螺仪测量的角速度随时间的积累值，因此算作传感器测量。

### 飞行器状态

* `b_v_x`: 在飞行器坐标系下的x轴方向速度。
* `b_v_y`: 在飞行器坐标系下的y轴方向速度。
* `b_v_z`: 在飞行器坐标系下的z轴方向速度。

### 任务相关状态

对速度控制任务，有如下额外状态：

* `next_target_g_v_x`: 任务设定的下一时刻目标速度在x轴的分量。
* `next_target_g_v_y`: 任务设定的下一时刻目标速度在y轴的分量。
* `next_target_g_v_z`: 任务设定的下一时刻目标速度在z轴的分量。

[gym]: https://gym.openai.com/
[map_example]: https://github.com/PaddlePaddle/RLSchool/blob/master/rlschool/quadrotor/default_map.txt
[default_sim_conf]: https://github.com/PaddlePaddle/RLSchool/blob/master/rlschool/quadrotor/config.json
