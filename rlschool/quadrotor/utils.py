#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np


def texture_coord(x, y, n=2):
    """Return the bounding vertices of the texture square.
    E.g. for texture.png, (0, 0) means the black-white tile texture.
    """
    m = 1. / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m


def texture_coords(top, bottom, side):
    """
    Return the texture squares for the six faces of the cube block to render.
    """
    top = texture_coord(*top)
    bottom = texture_coord(*bottom)
    side = texture_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result


SECTOR_SIZE = 8
TEXTURE_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'texture.png'))

TILE = texture_coords((0, 0), (0, 1), (0, 1))
WALL = texture_coords((1, 0), (1, 0), (1, 0))

FACES = [
    (0, 1, 0),
    (0, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
]


def sectorize(position):
    x, y, z = position
    x, y, z = int(round(x)), int(round(y)), int(round(z))
    x, y, z = x // SECTOR_SIZE, y // SECTOR_SIZE, z // SECTOR_SIZE
    return (x, 0, z)


def cube_vertices(position, n):
    """ Return the vertices of the cube at position with size 2*n.

    Note that in `pyglet.window.Window`, x-z plane is the ground plane.
    So here we unpack the position as `(x, z, y)` instead of `(x, y, z)`.

    """
    x, z, y = position

    return [
        # 4 vertices on top face
        x-n, y+n, z-n, x-n, y+n, z+n, x+n, y+n, z+n, x+n, y+n, z-n,
        # on bottom face
        x-n, y-n, z-n, x+n, y-n, z-n, x+n, y-n, z+n, x-n, y-n, z+n,
        # on left face
        x-n, y-n, z-n, x-n, y-n, z+n, x-n, y+n, z+n, x-n, y+n, z-n,
        # on right face
        x+n, y-n, z+n, x+n, y-n, z-n, x+n, y+n, z-n, x+n, y+n, z+n,
        # on front face
        x-n, y-n, z+n, x+n, y-n, z+n, x+n, y+n, z+n, x-n, y+n, z+n,
        # on back face
        x+n, y-n, z-n, x-n, y-n, z-n, x-n, y+n, z-n, x+n, y+n, z-n,
    ]


def geometry_hash(geometry):
    """
    Get an MD5 for a geometry object

    Parameters
    ------------
    geometry : object

    Returns
    ------------
    MD5 : str
    """
    if hasattr(geometry, 'md5'):
        # for most of our trimesh objects
        md5 = geometry.md5()
    elif hasattr(geometry, 'tostring'):
        # for unwrapped ndarray objects
        md5 = str(hash(geometry.tostring()))

    if hasattr(geometry, 'visual'):
        # if visual properties are defined
        md5 += str(geometry.visual.crc())
    return md5


def rotation_transform_mat(alpha, mode='yaw'):
    assert mode in ['yaw', 'pitch', 'roll']
    if mode == 'yaw':
        axis_0, axis_1 = 0, 1
    elif mode == 'pitch':
        axis_0, axis_1 = 0, 2
    elif mode == 'roll':
        axis_0, axis_1 = 1, 2

    transform = np.eye(4)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    transform[axis_0, axis_0] = cos_alpha
    transform[axis_1, axis_1] = cos_alpha

    if mode in ['yaw', 'roll']:
        transform[axis_0, axis_1] = -sin_alpha
        transform[axis_1, axis_0] = sin_alpha
    elif mode == 'pitch':
        transform[axis_0, axis_1] = sin_alpha
        transform[axis_1, axis_0] = -sin_alpha

    return transform
