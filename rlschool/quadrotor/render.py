import os
import math
import numpy as np
from collections import deque
import pyglet
from pyglet import image
from pyglet import gl
from pyglet.graphics import Batch, TextureGroup

from utils import TEXTURE_PATH, GRASS, SAND, BRICK, STONE, FACES, DRONE_FACE
from utils import sectorize, cube_vertices, drone_vertices


class Map(object):
    def __init__(self, map_config, horizon_view_size=8, init_drone_z=5):
        self.horizon_view_size = horizon_view_size

        # A Batch is a collection of vertex lists for batched rendering
        self.batch = Batch()

        # Manages an OpenGL texture
        self.group = TextureGroup(image.load(TEXTURE_PATH).get_texture())

        # A mapping from position to the texture for whole, global map
        self.whole_map = dict()

        # Same as `whole_map` but only contains the positions to show
        self.partial_map = dict()

        # A mapping from position to a pyglet `VertextList` in `partial_map`
        self._partial_map = dict()

        # A drone drawer, now it's just a flatten 1x1 face with SAND texture
        # TODO: load drone 3D model and render it
        self.drone_drawer = None

        # A mapping from sector to a list of positions (contiguous sub-region)
        # using sectors for fast rendering
        self.sectors = dict()

        # Use deque to populate calling of `_show_block` and `_hide_block`
        self.queue = deque()

        # Mark positions of bounding wall and obstacles in the map
        self._initialize(map_config, init_drone_z)

    def _initialize(self, map_config, init_drone_z):
        self.map_array, drone_yx = self._load_map_txt_config(map_config)
        self.drone_pos = [drone_yx[1], drone_yx[0], init_drone_z]
        h, w = self.map_array.shape
        for y in range(0, h):
            for x in range(0, w):
                assert self.map_array[y, x] >= 0, 'Error in map config'
                # Pave the floor
                for i in range(-4, 0):
                    self._add_block((x, y, i), SAND, immediate=False)
                self._add_block((x, y, 0), GRASS, immediate=False)

                if self.map_array[y, x] > 0:
                    texture = BRICK
                    if y in [0, h-1] or x in [0, w-1]:
                        texture = STONE
                    # Build the bounding wall or obstacles
                    for i in range(1, self.map_array[y, x]):
                        self._add_block((x, y, i), texture, immediate=False)

    def _load_map_txt_config(self, map_config):
        if not os.path.exists(map_config):
            raise ValueError('%s is an invalid txt config path.' % map_config)

        map_rows = []
        with open(map_config, 'r') as f:
            for line in f.readlines():
                map_rows.append([int(i) for i in line.split(' ')])

        map_arr = np.array(map_rows)

        # Extract the original quadrotor position
        drones_y, drones_x = np.where(map_arr == -1)
        assert len(drones_y) == 1   # Currently, only support single drone
        drone_y, drone_x = drones_y[0], drones_x[0]
        map_arr[drone_y, drone_x] = 0  # remove the drone marker

        return map_arr, (drone_y, drone_x)

    def _is_exposed(self, position):
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x+dx, y+dy, z+dz) not in self.whole_map:
                # At least one face is not covered by another cube block.
                return True
        return False

    def _add_block(self, position, texture, immediate=True):
        """ Add a block with the given `texture` and `position` to the world.

        Note that block is a 1x1x1 cube and its position is its centroid.

        Args:

        position (tuple): The (x, y, z) position of the block to add.
        texture (list): The coordinates of the texture squares, e.g. GRASS.
        immediate (bool): Whether or not to draw the block immediately.

        """
        if position in self.whole_map:
            # Not called for current static map
            assert False, 'Duplicated block!'
            self._remove_block(position, immediate)

        self.whole_map[position] = texture
        self.sectors.setdefault(sectorize(position), []).append(position)
        if immediate:
            if self._is_exposed(position):
                self.show_block(position)
            self._check_neighbors(position)

    def _remove_block(self, position, immediate=True):
        """ Remove the block at the given `position`.

        Args:

        position (tuple): The (x, y, z) position of the block to remove.
        immediate (bool): Whether or not to remove the block immediately.

        """
        del self.whole_map[position]
        self.sectors[sectorize(position)].remove(position)
        if immediate:
            if position in self.partial_map:
                self.hide_block(position)
            self._check_neighbors(position)

    def _check_neighbors(self, position):
        x, y, z = position
        for dx, dy, dz in FACES:
            pos = (x+dx, y+dy, z+dz)
            if pos not in self.whole_map:
                continue
            if self._is_exposed(pos):
                if pos not in self.partial_map:
                    self.show_block(pos)
            else:
                if pos in self.partial_map:
                    self.hide_block(pos)

    def _show_block(self, position, texture):
        vertex_data = cube_vertices(position, 0.5)  # 12x6=72
        texture_data = list(texture)  # 8x6=48
        vertex_count = len(vertex_data) // 3   # 24
        attributes = [
            ('v3f/static', vertex_data),
            ('t2f/static', texture_data)
        ]
        self._partial_map[position] = self.batch.add(
            vertex_count, gl.GL_QUADS, self.group, *attributes)

    def _hide_block(self, position):
        self._partial_map.pop(position).delete()

    def show_drone(self, position):
        if self.drone_drawer is not None:
            # NOTE: it may be costly to direct remove previous drone and redraw
            self.hide_drone()

        vertex_data = drone_vertices(position)
        texture_data = list(DRONE_FACE)
        vertex_count = len(vertex_data) // 3
        attributes = [
            ('v3f/static', vertex_data),
            ('t2f/static', texture_data)
        ]
        self.drone_drawer = self.batch.add(
            vertex_count, gl.GL_QUADS, self.group, *attributes)

    def hide_drone(self):
        self.drone_drawer.delete()
        self.drone_drawer = None

    def _enqueue(self, func, *args):
        self.queue.append((func, args))

    def _dequeue(self):
        func, args = self.queue.popleft()
        func(*args)

    def show_block(self, position, immediate=True):
        texture = self.whole_map[position]
        self.partial_map[position] = texture
        if immediate:
            self._show_block(position, texture)
        else:
            self._enqueue(self._show_block, position, texture)

    def hide_block(self, position, immediate=True):
        self.partial_map.pop(position)
        if immediate:
            self._hide_block(position)
        else:
            self._enqueue(self._hide_block, position)

    def show_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position not in self.partial_map and self._is_exposed(position):
                self.show_block(position, immediate=False)

    def hide_sector(self, sector):
        for position in self.sectors.get(sector, []):
            if position in self.partial_map:
                self.hide_block(position, immediate=False)

    def change_sectors(self, before, after):
        # Find the changed sectors and trigger show or hide operations

        # TODO: adjust the sector set when add extra view perspective
        # relative to the drone.
        # TODO: check the effect when manually resize the render window
        # FIXME: when the drone flies high, the green floor immediately
        # disappear
        before_set, after_set = set(), set()
        pad = self.horizon_view_size // 2
        for dx in range(-pad, pad+1):
            for dy in range(-pad, pad+1):
                dz = 0
                if dx ** 2 + dy ** 2 + dz ** 2 > (pad + 1) ** 2:
                    continue
                if before:
                    x, y, z = before
                    before_set.add((x+dx, y+dy, z+dz))
                if after:
                    x, y, z = after
                    after_set.add((x+dx, y+dy, z+dz))

        show = after_set - before_set
        hide = before_set - after_set
        for sector in show:
            self.show_sector(sector)
        for sector in hide:
            self.hide_sector(sector)

    def process_queue(self):
        # NOTE: no scheduled interval timer, we render by manually calling
        # `RenderWindow.view()`. So we process queue without time contrains.
        # In other words, it's a copy of `process_entire_queue()`
        while self.queue:
            self._dequeue()

    def process_entire_queue(self):
        while self.queue:
            self._dequeue()


class RenderWindow(pyglet.window.Window):
    def __init__(self,
                 map_config,
                 horizon_view_size=8,
                 init_drone_z=5,
                 perspective_fovy=65.,
                 perspective_aspect=4/3.,  # i.e. 800/600
                 perspective_zNear=0.1,
                 perspective_zFar=60.,
                 sky_rgba=(0.5, 0.69, 1.0, 1),
                 width=800, height=600,
                 caption='quadrotor',
                 resizable=False):
        super(RenderWindow, self).__init__(
            width=width, height=height, caption=caption, resizable=resizable)

        self.internal_map = Map(
            map_config,
            horizon_view_size=horizon_view_size,
            init_drone_z=init_drone_z)

        # The label to display in the top-left of the canvas
        self.label = pyglet.text.Label(
            '', font_name='Arial', font_size=18, x=10, y=self.height - 10,
            anchor_x='left', anchor_y='top', color=(0, 0, 0, 255))

        # Current (x, y, z) position of the drone in the world,
        # specified with floats.
        self.position = tuple([float(i) for i in self.internal_map.drone_pos])

        # (vertical plane rotation, horizontal rotation)
        # vertical rotation: [-90, 90], horizontal rotation unbounded
        # TODO: update the rotation according the drone initial pose
        self.rotation = (-30, 0)

        # Config perspective
        self.perspective = [perspective_fovy, perspective_aspect,
                            perspective_zNear, perspective_zFar]

        self.sector = None

        gl.glClearColor(*sky_rgba)
        self.set_visible()

    def update(self, dt):
        self.internal_map.process_queue()
        sector = sectorize(self.position)
        if sector != self.sector:
            self.internal_map.change_sectors(self.sector, sector)
            if self.sector is None:
                self.internal_map.process_entire_queue()
            self.sector = sector

        self.internal_map.show_drone(self.position)

    def view(self, drone_state, dt):
        # TODO: support to udpate the view according to the pose of the drone
        self.position = (drone_state['x'], drone_state['y'], drone_state['z'])

        # Actually, `dt` does not work now, as we update the state in env.py
        self.update(dt)

        self.clear()
        self._setup_3d()
        gl.glColor3d(1, 1, 1)
        self.internal_map.batch.draw()
        self._setup_2d()
        self._draw_label()

        self.dispatch_events()
        self.flip()

    def _setup_2d(self):
        w, h = self.get_size()
        gl.glDisable(gl.GL_DEPTH_TEST)

        viewport = self.get_viewport_size()
        gl.glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, max(1, w), 0, max(1, h), -1, 1)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def _setup_3d(self):
        w, h = self.get_size()
        gl.glEnable(gl.GL_DEPTH_TEST)

        viewport = self.get_viewport_size()
        gl.glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(*self.perspective)

        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        y, x = self.rotation
        gl.glRotatef(x, 0, 1, 0)
        gl.glRotatef(-y, math.cos(math.radians(x)),
                     0, math.sin(math.radians(x)))
        # NOTE: for GL render, its x-z plane is the ground plane,
        # so we unpack the position using `(x, z, y)` instead of `(x, y, z)`

        # TODO: add these extra perspective tune to keyword args of init func
        x, z, y = self.position
        z += 1.3
        y += 0.8
        gl.glTranslatef(-x, -y, -z)

    def _draw_label(self):
        self.label.text = 'xyz: (%.2f, %.2f, %.2f)' % self.position
        self.label.draw()
