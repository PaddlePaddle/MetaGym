# ElevatorSimulator

ElevatorSimulator是一个电梯调度模拟环境

<img src="demo_image.gif" width="400"/>



## 下载

可以通过pip下载：

```python
pip install RLschool
```


## 基本接口

类似gym，ElevatorSimulator提供了三个基本接口：

- reset(self)：重置环境，返回observation。
- step(self, action)：根据action调整环境，返回observation，reward，done，info。
- render(self)：显示一个timestep内的环境。

## 运行逻辑

电梯负责处理电梯内乘客按下的楼层（target_floot），依次停靠。Dispatcher负责调配电梯到有人等待的楼层去接乘客（dispatch_target）以及分配接到人后的方向（dispatch_target_direction）。

电梯若无法在dispatch_target停下，则忽略dispatch_target。若希望电梯在某一层停下，则需要在电梯停在该楼层之前，保持dispatch_target一直为该楼层，或返回ElevatorAction中TargetFloor为-1。

dispatch_target_direction负责指示接到乘客后电梯行驶方向，当电梯内无人，电梯静止且无方向时有效。

## 示例

我们提供了电梯调度算法的[示例][demo]，包括强化学习和规则，以供参考。


[demo]: https://github.com/Banmahhhh/RLSchool/blob/master/liftsim/demo.py
