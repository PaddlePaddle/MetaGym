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

import setuptools

setuptools.setup(
    name="rlschool",  # pypi中的名称，pip或者easy_install安装时使用的名称
    version="1.0",
    author="",
    author_email="",
    # download_url
    description=("Reinforcement learning simulators"),
    license="GPLv3",
    keywords="redis subscripe",
    url="",
    packages=['liftsim/environment'],  # 需要打包的目录列表
    install_requires=['pyglet>=1.4.1'],
    # classifiers
)
