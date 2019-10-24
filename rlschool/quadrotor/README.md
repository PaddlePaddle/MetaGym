# Quadrotor

## Dependencies

**C++ dependencies**:
* Eigen3
* Boost

**Python dependencies**:
* Numpy
* Pyglet v1.4 (or less than v1.4)

Note that current version only supports Python 2.7.

## Build

Build with cmake by following commands:

```sh
cd uranusim
mkdir build && cd build
cmake ..
make -j 8
```

Note that if you use anaconda, you need to manually set the variables `PYTHON_LIBRARY`, for example:

```sh
export CONDA_HOME=~/anaconda3
mkdir build && cd build
cmake -D PYTHON_LIBRARY=$CONDA_HOME/envs/rlschool-py27/lib/libpython2.7.dylib \
      -D PYTHON_INCLUDE_DIR=$CONDA_HOME/envs/rlschool-py27/include/python2.7 ..
```
