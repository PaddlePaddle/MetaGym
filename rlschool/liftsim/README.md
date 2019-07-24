# LiftSim

LiftSim是一个电梯调度模拟环境

<img src="demo_image.gif" width="400"/>



## 下载

可以通过pip下载：

```python
pip install rlschool
```


## 基本接口

类似gym，liftsim提供了三个基本接口：

- reset(self)：重置环境，返回observation。
- step(self, action)：根据action调整环境，返回observation，reward，done，info。
- render(self)：显示一个timestep内的环境。

```python
from rlschool import LiftSim
from rlschool import ElevatorAction

env = LiftSim()
observation = env.reset()
action = [ElevatorAction(0, 1) for i in range(4)]
for i in range(100):
    next_obs, reward, done, info = env.step(action)
```

## MansionState/ElevatorState

reset(self)和step(self, action)返回一个namedtuple，MansionState。

- MansionState = collections.namedtuple("MansionState", 
                    ["ElevatorStates", "RequiringUpwardFloors","RequiringDownwardFloors"])

    ELevatorState为各个电梯的情况。RequiringUpwardFloors是list，包括有人等待向上的楼层；RequiringDownwardFloors包括有人等待向下的楼层。
- ElevatorStates = collections.namedtuple("ElevatorState",
                   ["Floor", "MaximumFloor",
                   "Velocity", "MaximumSpeed",
                   "Direction", "DoorState",
                   "CurrentDispatchTarget", "DispatchTargetDirection",
                   "LoadWeight", "MaximumLoad",
                   "ReservedTargetFloors", "OverloadedAlarm",
                   "DoorIsOpening", "DoorIsClosing"])

    Floor：电梯当前楼层；   MaximumFloor：大楼最高楼层；    Velocity：电梯当前速度；    MaximumSpeed：电梯最大速度；Direction：电梯方向（-1为向下，1为向上，0为无方向）；   DoorState：电梯门当前打开的比例；   CurrentDispatchTarget：电梯当前dispatch到的目标楼层；   DispatchTargetDirection：电梯dispatch到的方向；     LoadWeight：电梯承载的质量；        MaximumLoad：电梯最大能承载的质量；     ReservedTargetFloors：list，存储电梯内乘客的目标楼层；      OverloadedAlarm：指示电梯是否超载；     DoorIsOpening：指示电梯门是否正在打开；     DoorIsClosing：指示电梯门是否正在关闭。

## 运行逻辑

电梯负责处理电梯内乘客按下的楼层（target_floot），依次停靠。Dispatcher负责调配电梯到有人等待的楼层去接乘客（dispatch_target）以及分配接到人后的方向（dispatch_target_direction）。

电梯若无法在dispatch_target停下，则忽略dispatch_target。若希望电梯在某一层停下，则需要在电梯停在该楼层之前，保持dispatch_target一直为该楼层，或返回ElevatorAction中TargetFloor为-1。

dispatch_target_direction负责指示接到乘客后电梯行驶方向，当电梯内无人，电梯静止且无方向时有效。

## 示例

我们提供了电梯调度算法的[示例][demo]，包括强化学习和规则，以供参考。

## 评价标准

根据电梯内乘客等待时长、电梯外人们排队时长、电梯消耗的能量、放弃的人数（排队的人五分钟后自动放弃）来计算reward。公式：
$$- (time\_consume + 0.01 * energy\_consume + 1000 * given\_up\_persons) * 1.0e-5$$

## 提交

在[此处][submit]提交结果


[demo]: https://github.com/Banmahhhh/RLSchool/blob/master/liftsim/demo.py
[submit]: https://www.google.com/
