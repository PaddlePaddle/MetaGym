# Quadrotor

## Dependencies

**C++ dependencies**:
* Eigen3
* Boost

**Python dependencies**:
* Numpy
* Pyglet v1.4 (or less than v1.4)
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
python -m rlschool.quadrotor.env
```
