import os
import math
import numpy as np
from collections import deque
import trimesh
from trimesh import rendering
import pyglet
from pyglet import image
from pyglet import gl
from pyglet.graphics import Batch, TextureGroup

from rlschool.quadrotor.utils import TEXTURE_PATH, TILE, FACES
from rlschool.quadrotor.utils import sectorize, cube_vertices, geometry_hash, \
    rotation_transform_mat


class Map(object):
    def __init__(self, drone_3d_model, horizon_view_size=8, init_drone_z=5):
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

        # A mapping from sector to a list of positions (contiguous sub-region)
        # using sectors for fast rendering
        self.sectors = dict()

        # Use deque to populate calling of `_show_block` and `_hide_block`
        self.queue = deque()

        # A graphics batch to draw drone 3D model
        self.drone_batch = pyglet.graphics.Batch()
        # Load drone triangular mesh and scene
        self.drone_name = os.path.basename(drone_3d_model)
        self.drone_mesh = trimesh.load(drone_3d_model)
        self.drone_scene = self.drone_mesh.scene()
        # Drawer stores drone scene geometry as vertex list in its model space
        self.drone_drawer = None
        # Store drone geometry hashes for easy retrival
        self.drone_vertex_list_hash = ''
        # Store drone geometry rendering mode, default gl.GL_TRIANGLES
        self.drone_vertex_list_mode = gl.GL_TRIANGLES
        # Store drone geometry texture
        self.drone_texture = None

        color = np.array([0, 0, 0, 255], dtype=np.uint8)
        for facet in self.drone_mesh.facets:
            self.drone_mesh.visual.face_colors[facet] = color
            # TODO: figure how to paint the drone pretty
            # self.drone_mesh.visual.face_colors[facet] = \
            #     trimesh.visual.random_color()

        # Mark positions of bounding wall and obstacles in the map
        self._initialize(init_drone_z)

    def _initialize(self, init_drone_z):
        h = w = 100
        self.drone_pos = [h // 2, w // 2, init_drone_z]
        for y in range(0, h):
            for x in range(0, w):
                # Pave the floor
                self._add_block((x, y, 0), TILE, immediate=False)

        self._add_drone()

    def _is_exposed(self, position):
        x, y, z = position
        for dx, dy, dz in FACES:
            if (x+dx, y+dy, z+dz) not in self.whole_map:
                # At least one face is not covered by another cube block.
                return True
        return False

    def _add_drone(self):
        """ Add the drone 3D model in its own model space.
        """
        for name, geom in self.drone_scene.geometry.items():
            if geom.is_empty:
                continue
            if geometry_hash(geom) == self.drone_vertex_list_hash:
                continue

            if name == self.drone_name:
                args = rendering.convert_to_vertexlist(geom, smooth=True)
                self.drone_drawer = self.drone_batch.add_indexed(*args)
                self.drone_vertex_list_hash = geometry_hash(geom)
                self.drone_vertex_list_mode = args[1]

                try:
                    assert len(geom.visual.uv) == len(geom.vertices)
                    has_texture = True
                except BaseException:
                    has_texture = False

                if has_texture:
                    self.drone_texture = rendering.material_to_texture(
                        geom.visual.material)

    def _add_block(self, position, texture, immediate=True):
        """ Add a block with the given `texture` and `position` to the world.

        Note that block is a 1x1x1 cube and its position is its centroid.

        Args:

        position (tuple): The (x, y, z) position of the block to add.
        texture (list): The coordinates of the texture squares, e.g. TILE.
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

    def _enqueue(self, func, *args):
        self.queue.append((func, args))

    def _dequeue(self):
        func, args = self.queue.popleft()
        func(*args)

    def show_drone(self, position, rotation):
        # Get the transform matrix for drone 3D model
        # TODO: support to render the rotation pose of the drone
        x, z, y = position
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]

        # NOTE: change the view size of drone 3D model
        transform[0, 0] = 2.5
        transform[1, 1] = 2.5
        transform[2, 2] = 2.5

        yaw, pitch, roll = rotation
        transform = np.dot(transform, rotation_transform_mat(yaw, 'yaw'))
        transform = np.dot(transform, rotation_transform_mat(pitch, 'pitch'))
        transform = np.dot(transform, rotation_transform_mat(roll, 'roll'))

        # Add a new matrix to the model stack to transform the model
        gl.glPushMatrix()
        gl.glMultMatrixf(rendering.matrix_to_gl(transform))

        # Enable the target texture
        if self.drone_texture is not None:
            gl.glEnable(self.drone_texture.target)
            gl.glBindTexture(self.drone_texture.target, self.drone_texture.id)

        # Draw the mesh with its transform applied
        self.drone_drawer.draw(mode=self.drone_vertex_list_mode)
        gl.glPopMatrix()

        # Disable texture after using
        if self.drone_texture is not None:
            gl.glDisable(self.drone_texture.target)

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
                 drone_3d_model=None,
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
        if drone_3d_model is None:
            this_dir = os.path.realpath(os.path.dirname(__file__))
            drone_3d_model = os.path.join(this_dir, 'quadcopter.stl')

        super(RenderWindow, self).__init__(
            width=width, height=height, caption=caption, resizable=resizable)

        self.internal_map = Map(
            drone_3d_model,
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

        self._gl_set_background(sky_rgba)
        self._gl_enable_color_material()
        self._gl_enable_blending()
        self._gl_enable_smooth_lines()
        self._gl_enable_lighting(self.internal_map.drone_scene)
        self.set_visible()

    def update(self, dt):
        self.internal_map.process_queue()
        sector = sectorize(self.position)
        if sector != self.sector:
            self.internal_map.change_sectors(self.sector, sector)
            if self.sector is None:
                self.internal_map.process_entire_queue()
            self.sector = sector

    def view(self, drone_state, dt):
        self.position = (drone_state['x'], drone_state['y'], drone_state['z'])
        rot = (drone_state['yaw'], drone_state['pitch'], drone_state['roll'])

        # Actually, `dt` does not work now, as we update the state in env.py
        self.update(dt)

        self.clear()
        self._setup_3d()
        gl.glColor3d(1, 1, 1)
        self.internal_map.batch.draw()
        self.internal_map.show_drone(self.position, rot)
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
        gl.glDepthFunc(gl.GL_LEQUAL)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

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
        # z += 1.3
        # y += 0.8
        y += 3
        z += 3
        gl.glTranslatef(-x, -y, -z)

    def _draw_label(self):
        self.label.text = 'xyz: (%.2f, %.2f, %.2f)' % self.position
        self.label.draw()

    @staticmethod
    def _gl_set_background(background):
        gl.glClearColor(*background)

    @staticmethod
    def _gl_unset_background():
        gl.glClearColor(*[0, 0, 0, 0])

    @staticmethod
    def _gl_enable_color_material():
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK,
                           gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_AMBIENT,
                        rendering.vector_to_gl(
                            0.192250, 0.192250, 0.192250))
        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_DIFFUSE,
                        rendering.vector_to_gl(
                            0.507540, 0.507540, 0.507540))
        gl.glMaterialfv(gl.GL_FRONT,
                        gl.GL_SPECULAR,
                        rendering.vector_to_gl(
                            .5082730, .5082730, .5082730))

        gl.glMaterialf(gl.GL_FRONT,
                       gl.GL_SHININESS,
                       .4 * 128.0)

    @staticmethod
    def _gl_enable_blending():
        # enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA,
                       gl.GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _gl_enable_smooth_lines():
        # make the lines from Path3D objects less ugly
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        # set the width of lines to 4 pixels
        gl.glLineWidth(4)
        # set PointCloud markers to 4 pixels in size
        gl.glPointSize(4)

    @staticmethod
    def _gl_enable_lighting(scene):
        """
        Take the lights defined in scene.lights and
        apply them as openGL lights.
        """
        gl.glEnable(gl.GL_LIGHTING)
        # opengl only supports 7 lights?
        for i, light in enumerate(scene.lights[:7]):
            # the index of which light we have
            lightN = eval('gl.GL_LIGHT{}'.format(i))

            # get the transform for the light by name
            matrix = scene.graph.get(light.name)[0]

            # convert light object to glLightfv calls
            multiargs = rendering.light_to_gl(
                light=light,
                transform=matrix,
                lightN=lightN)

            # enable the light in question
            gl.glEnable(lightN)
            # run the glLightfv calls
            for args in multiargs:
                gl.glLightfv(*args)
