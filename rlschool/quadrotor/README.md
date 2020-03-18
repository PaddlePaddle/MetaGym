# Quadrotor

## Dependencies

**C++ dependencies**:
* Eigen3
* Boost

For Ubuntu, you can install them using following commands:

```sh
sudo apt-get install libeigen3-dev libboost-all-dev
```

For MacOS, use commands:

```sh
brew install boost eigen
```

**Python dependencies**:
* Numpy
* Pyglet
* Trimesh (for loading stl model file)

## Install

For local installation, execute following commands:

```sh
git clone https://github.com/PaddlePaddle/RLSchool
cd RLSchool
pip install .
```

## Test Visualization

```sh
# For 'no_collision' task
python -m rlschool.quadrotor.env

# For 'velocity_control' task
python -m rlschool.quadrotor.env velocity_control
```

* "no_collision" task

<img src="demo/demo_no_collision.gif" width="400"/>

* "velocity_control" task

<img src="demo/demo_velocity_control.gif" width="400"/>

Yellow arrow is the expected velocity vector; orange arrow is the real velocity vector.

## Example

Quadrotor environment follows the standard [gym][gym] APIs to create, run, render and close an environment. Currently, Quadrotor supports two tasks: "no_collision" and "velocity_control".

To create a quadrotor environment for "no_collision" task:

```python
from rlschool import make_env
env = make_env("Quadrotor", task="no_collision", map_file=None, simulator_conf=None)
env.reset()
```

When the argument `map_file` is `None`, the world is a 100x100 flatten floor. Once the drone hits the floor, the environment episode terminates. The map file is a text file like [default_map.txt][map_example], in which each number represents the height of obstacle wall in corresponding location. `-1` marks the initial location of the drone. If needed, you can set `map_file` to the path of a custom map file.

When the argument `simulator_conf` is `None`, the environment would use the default simulator configuration [config.xml][default_sim_conf]. If you want to change the dynamics of the drone, you need to create a new _config.xml_, and set `simulator_conf` to the path of the new _config.xml_.

To create a quadrotor environment for "velocity_control" task:

```python
from rlschool import make_env
env = make_env("Quadrotor", task="velocity_control", seed=0, simulator_conf=None)
env.reset()
```

The argument `seed` is used for sampling a trajectory of expected velocity vectors. Its default value is `0`.

### Action

The action for Quadrotor environment is the respective voltage value of four propeller motors. Each voltage value is in range `[0.10, 15.0]`, by default, which is set in [config.xml](default_sim_conf).

As you expected, action `[1.0, 1.0, 1.0, 1.0]` would lead straight top-down movement. Please check the visualization using following code:

```python
from rlschool import make_env

env = make_env("Quadrotor", task="no_collision")
env.reset()
env.render()

reset = False
while not reset:
    state, reward, reset = env.step([1.0, 1.0, 1.0, 1.0])
    env.render()
```

### State

The state of the drone belongs to three categories: sensor measurements, flighting state and task related state.

#### Sensor Measurements

* `acc_x`: the measurement of accelerometer in x direction.
* `acc_y`: the measurement of accelerometer in y direction.
* `acc_z`: the measurement of accelerometer in z direction.
* `gyro_x`: the measurement of gyroscope in x direction.
* `gyro_y`: the measurement of gyroscope in y direction.
* `gyro_z`: the measurement of gyroscope in z direction.

#### Flighting State

* `x`: position of the drone in x direction.
* `y`: position of the drone in y direction.
* `z`: position of the drone in z direction.
* `g_v_x`: velocity of the drone in x direction, in global coordinate.
* `g_v_y`: velocity of the drone in y direction, in global coordinate.
* `g_v_z`: velocity of the drone in z direction, in global coordinate.
* `b_v_x`: velocity of the drone in x direction, in its own body coordinate.
* `b_v_y`: velocity of the drone in y direction, in its own body coordinate.
* `b_v_z`: velocity of the drone in z direction, in its own body coordinate.
* `w_x`: body rotation angular speed of the drone in x direction.
* `w_y`: body rotation angular speed of the drone in y direction.
* `w_z`: body rotation angular speed of the drone in z direction.
* `pitch`: body rotation angular around the x axis.
* `roll`: body rotation angular around the y axis.
* `yaw`: body rotation angular around the z axis.
* `power`: total power of four propeller motors.

#### Task Related State

For "velocity_control" task, it has:

* `next_target_g_v_x`: next expected velocity of the drone in x direction, in global coordinate.
* `next_target_g_v_y`: next expected velocity of the drone in y direction, in global coordinate.
* `next_target_g_v_z`: next expected velocity of the drone in z direction, in global coordinate.

### Simulator Configuration

This part explains physical parameters in [config.xml][default_sim_conf].

* <img src="https://render.githubusercontent.com/render/math?math=m">: `simulator.quality`, the whole quality of the drone.
* <img src="https://render.githubusercontent.com/render/math?math=\mathbf{J}">: `simulator.inertia`, the 3x3 **symmetric** inertia matrix (only require values of `xx`, `xy`, `xz`, `yy`, `yz`, and `zz`).

>**NOTE**: Considering the formula of inertia, value of inertia should match the magnitude of `simulator.quality` * (`simulator.propeller.1.x` ^ 2 + `simulator.propeller.1.y` ^ 2).

* <img src="https://render.githubusercontent.com/render/math?math=\mathbf{F}_d">: the drag force.
  * its direction is inverse to the velocity vector.
  * its three orthogonal component forces are `simulator.drag.f_xx`, `simulator.drag.f_yy` and `simulator.drag.f_zz`.
* <img src="https://render.githubusercontent.com/render/math?math=\mathbf{M}_d">: the drag moment.
  * its three components are `simulator.drag.m_xx`, `simulator.drag.m_yy` and `simulator.drag.m_zz`.
* <img src="https://render.githubusercontent.com/render/math?math=o">: `simulator.gravity_center`, the gravity center (`x`, `y`, `z`).
* <img src="https://render.githubusercontent.com/render/math?math=M_m">: `simulator.thrust.Mm`, the mechanical torque of the motor along propeller axis.
* <img src="https://render.githubusercontent.com/render/math?math=J_M">: `simulator.thrust.Jm`, the motor inertia along propeller axis.
* <img src="https://render.githubusercontent.com/render/math?math=R_A">: `simulator.thrust.RA`, the anchor resistance of the motor.
* <img src="https://render.githubusercontent.com/render/math?math=\Phi">: `simulator.thrust.phi`, the coefficient for converting angular speed of propeller (<img src="https://render.githubusercontent.com/render/math?math=\omega_M">) to anchor voltage (<img src="https://render.githubusercontent.com/render/math?math=U_A">).
* <img src="https://render.githubusercontent.com/render/math?math=C_{T, i}, i=0,1,2">: `simulator.thrust.CT`, the coefficients of the formula for calculating thrust force from given angular speed of propeller (<img src="https://render.githubusercontent.com/render/math?math=\omega_M">) and ascending velocity of propeller (<img src="https://render.githubusercontent.com/render/math?math=v_1">). See Eq 1.
* <img src="https://render.githubusercontent.com/render/math?math=dt">: `simulator.precision`, the minimum time precision to run the dynamics model.
* `simulator.propeller`: the positions of four propellers.
* `simulator.fail.velocity`: the maximum velocity allowed in global coordinate.
* `simulator.fail.w`: the maximum body angular velocity allowed.
* `simulator.fail.range`: the maximum range allowed to reach.
* `simulator.electric`: the minimum and maximum voltage for each motor.

>*Eq 1 (Thrust)*:<br>
><img src="https://render.githubusercontent.com/render/math?math=T = C_{T,0} \omega_M^2 %2B C_{T,1} v_1 \omega_M %2B \text{sign}(v_1) C_{T,2} v_1^2">

[gym]: https://gym.openai.com/
[map_example]: https://github.com/PaddlePaddle/RLSchool/blob/master/rlschool/quadrotor/default_map.txt
[default_sim_conf]: https://github.com/PaddlePaddle/RLSchool/blob/master/rlschool/quadrotor/quadrotorsim/config.xml
