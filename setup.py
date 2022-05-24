#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
from setuptools import setup, find_packages

__version__ = '0.1.0'

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='metagym',
    version=__version__,
    author='parl_dev',
    author_email='',
    description=('MetaGym: environments for benchmarking Reinforcement Learning and Meta Reinforcement Learning'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/MetaGym',
    license="GPLv3",
    packages=[package for package in find_packages()
              if package.startswith('metagym')],
    package_data={'metagym': [
        './liftsim/config.ini',
        './liftsim/environment/animation/resources/*.png',
        './liftsim/environment/mansion/person_generators/mansion_flow.npy',
        './quadrotor/quadcopter.stl',
        './quadrotor/texture.png',
        './quadrotor/config.json']
    },
    tests_require=['pytest', 'mock'],
    include_package_data=True,
    install_requires=[
        'gym>=0.18.0',
        'numpy>=1.16.4',
        'Pillow>=6.2.2',
        'six>=1.12.0',
    ],
    extras_require={
        'metamaze': ['pygame>=2.0.2dev2', 'numba>=0.54.0'],
        'quadrupedal': ['scipy>=0.12.0', 'pybullet>=3.0.7', 'attrs>=20.3.0'],
        'quadrotor': ['scipy>=0.12.0', 'networkx>=2.2', 'trimesh>=3.2.39', 'networkx>=2.2', 'colour>=0.1.5'],
        'liftsim': ['configparser>=3.7.4', 'pyglet==1.5.0; python_version>="3"', 'pyglet==1.4.0; python_version<"3"'],
        'navigator2d': ['pygame>=2.0.2dev2']
    },
    zip_safe=False,
)
