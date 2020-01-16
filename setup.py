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

import os
import re
import io
import sys
import platform
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                'CMake must be installed to build the following '
                'extensions: ' + ', '.join(e.name for e in self.extensions))

        if platform.system() == 'Windows':
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError('CMake >= 3.1.0 is required on Windows')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == 'Windows':
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='rlschool',
    version='0.2.0',
    author='parl_dev',
    author_email='',
    description=('RLSchool: Excellent environments for reinforcement Learning benchmarking'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/RLSchool',
    license="GPLv3",
    packages=[package for package in find_packages()
              if package.startswith('rlschool')],
    package_data={'rlschool': [
        './liftsim/config.ini',
        './liftsim/environment/animation/resources/*.png',
        './liftsim/environment/mansion/person_generators/mansion_flow.npy',
        './quadrotor/quadcopter.stl',
        './quadrotor/texture.png',
        './quadrotor/quadrotorsim/config.xml']
    },
    tests_require=['pytest', 'mock'],
    include_package_data=True,
    install_requires=[
        'pyglet>=1.2.0,<=1.4.0',
        'six>=1.12.0',
        'numpy>=1.16.4',
        'configparser>=3.7.4',
        'trimesh>=3.2.39',
        'networkx>=2.2',
        'colour>=0.1.5',
        'scipy>=0.12.0'
    ],
    ext_modules=[CMakeExtension(
        'quadrotorsim', './rlschool/quadrotor/quadrotorsim')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
