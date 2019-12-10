# Quadrotor

## Dependencies

**C++ dependencies**:
* Eigen3
* Boost

**Python dependencies**:
* Numpy
* Pyglet
* Trimesh (for loading stl model file)

## Install

For local installation, execute following commands:

```sh
git clone --recursive https://github.com/PaddlePaddle/RLSchool
cd RLSchool
pip install .
```

Note that this simulator has been tested in Python 2.7 and Python 3.6 environments.

## Test Visualization

```sh
# For 'no_collision' task
python -m rlschool.quadrotor.env

# For 'velocity_control' task
python -m rlschool.quadrotor.env velocity_control
```
