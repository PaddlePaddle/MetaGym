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

from setuptools import setup, find_packages
import os
import re
import io

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="rlschool",  # pypi中的名称，pip或者easy_install安装时使用的名称
    version="0.1.1",
    author="",
    author_email="",
    description=("RLSchool: Excellent environments for reinforcement Learning benchmarking"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/RLSchool',
    license="GPLv3",
    keywords="redis subscripe",
    packages = [package for package in find_packages()
                if package.startswith('rlschool')],
    package_data={'rlschool':[
        './liftsim/config.ini', 
        './liftsim/environment/animation/resources/*.png',
        './liftsim/environment/mansion/person_generators/mansion_flow.npy'],
    },
    tests_require=['pytest', 'mock'],
    include_package_data=True,
    install_requires=[
        'pyglet>=1.2.0',
        'six>=1.12.0',
        'numpy>=1.16.4',
        'configparser>=3.7.4',
    ],
)
