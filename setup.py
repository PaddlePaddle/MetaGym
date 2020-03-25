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
import io
import sys
import setuptools
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

__version__ = '0.2.0'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class find_system_cpp_include(object):
    """Helper class to find C++ dependencies include path from system.

    For RLSchool, this class helps to find include path for Boost and Eigen3.
    """

    def __init__(self, name='boost', hint=None, with_name=False):
        self.name = name
        self.include_path = None

        search_dirs = [] if hint is None else hint
        search_dirs.extend([
            "/usr/local/include",
            "/usr/local/homebrew/include",
            "/opt/local/var/macports/software",
            "/opt/local/include",
            "/usr/include",
            "/usr/include/local"
        ])

        for d in search_dirs:
            path = os.path.join(d, name)
            if os.path.exists(path):
                if with_name:
                    self.include_path = path
                else:
                    self.include_path = d
                break

        if self.include_path is None:
            raise RuntimeError('Cannot find include_path for %s' % self.name)

    def __str__(self):
        return self.include_path


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++11', '-std=c++14', '-std=c++17']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')



with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

ext_modules, cpp_deps_errors = [], None
try:
    ext_modules = [
        Extension(
            'quadrotorsim',
            ['rlschool/quadrotor/quadrotorsim/src/simulator.cpp'],
            include_dirs=[
                # Path to Boost headers
                find_system_cpp_include(name='boost', with_name=False),
                # Path to Eigen3 headers
                find_system_cpp_include(name='eigen3', with_name=True),
                # Path to pybind11 headers
                get_pybind_include(),
                get_pybind_include(user=True),
                # Path to quadrotorsim headers
                'rlschool/quadrotor/quadrotorsim/include'
            ],
            language='c++'
        )
    ]
except RuntimeError as e:
    cpp_deps_errors = str(e)

# Force pip to install pybind11 before building extension
if len(ext_modules) > 0:
    setup_requires = ['pybind11>=2.4']
    os.system('{} -m pip install {}'.format(
        sys.executable, ' '.join(setup_requires)))


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        distribution_ver = self.distribution.get_version()
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % distribution_ver)
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % distribution_ver)
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


class PostInstall(install):
    def run(self):
        install.run(self)
        if cpp_deps_errors is not None:
            # Only show this with `-v' arg
            print('[WARNING] {}'.format(cpp_deps_errors))
            print('Failed to install environments with cpp extensions.')


setup(
    name='rlschool',
    version=__version__,
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
        'pyglet==1.5.0; python_version>="3"',
        'pyglet==1.4.0; python_version<"3"',
        'Pillow>=6.2.2',
        'six>=1.12.0',
        'numpy>=1.16.4',
        'configparser>=3.7.4',
        'trimesh>=3.2.39',
        'networkx>=2.2',
        'colour>=0.1.5',
        'scipy>=0.12.0'
    ],
    ext_modules=ext_modules,
    cmdclass=dict(build_ext=BuildExt, install=PostInstall),
    zip_safe=False,
)
