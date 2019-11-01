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
TEXTURE_PATH = 'texture.png'

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


def drone_vertices(position):
    """ Return the vertices of the drone (1x1 flatten plane) at position.

    Note that in `pyglet.window.Window`, x-z plane is the ground plane.
    So here we unpack the position as `(x, z, y)` instead of `(x, y, z)`.

    TODO: support to load .stl 3D model and extract Delaunay triangulation.
    """
    x, z, y = position
    n = 0.5
    return [x-n, y, z-n, x-n, y, z+n, x+n, y, z+n, x+n, y, z-n]


DRONE_FACE = texture_coord(1, 1)
