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

"""
Setup script.

Authors: wangfan04(wangfan04@baidu.com)
Date:    2019/05/22 19:30:16
"""

from setuptools import setup, find_packages
import os
import re

setup(
    name="rlschool",  # pypi中的名称，pip或者easy_install安装时使用的名称
    version="1.0",
    author="",
    author_email="",
    description=("A reinforcement learning simulators"),
    license="GPLv3",
    keywords="redis subscripe",
    # url="",
    # packages=_find_packages('rlschool'),
    packages = [package for package in find_packages()
                if package.startswith('rlschool')],
    package_data={'rlschool':['./liftsim/config.ini', './liftsim/environment/animation/resources/*.png']},
    include_package_data=True,
    install_requires=[
        'pyglet>=1.2.0',
        'six>=1.12.0',
        'numpy>=1.16.4',
        'configparser>=3.7.4',
    ],
)
