cmake_minimum_required(VERSION 2.8.12)
project(uranusim)

# find_package(PythonLibs 2.7 REQUIRED)
# include_directories(${PYTHON_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
find_package(Boost REQUIRED COMPONENTS)
include_directories(${Boost_INCLUDE_DIRS})

if(CMAKE_COMPILER_IS_GNUCXX)
    message(STATUS "optional:-std=c++11")
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
endif(CMAKE_COMPILER_IS_GNUCXX)

MESSAGE( STATUS "Buiding uranus simulator Library...")
INCLUDE_DIRECTORIES(./include)

if(APPLE)
set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif(APPLE)

add_subdirectory(pybind11)
pybind11_add_module(uranusim src/uranus_simulator.cpp)
