#!/usr/bin/env python
# -*- coding: UTF-8 -*-
##########################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
##########################################################################
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
    package_data={'rlschool':['./liftsim/config.ini']},
    include_package_data=True,
    install_requires=['pyglet>=1.2.0'],
)
