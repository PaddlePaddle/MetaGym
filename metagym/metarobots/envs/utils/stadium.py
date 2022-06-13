import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)

from .scene_bases import Scene
import pybullet

class StadiumScene(Scene):
	multiplayer = False
	zero_at_running_strip_start_line = True   # if False, center of coordinates (0,0,0) will be at the middle of the stadium
	stadium_halflen   = 105 * 0.25	# FOOBALL_FIELD_HALFLEN
	stadium_halfwidth = 50 * 0.25	 # FOOBALL_FIELD_HALFWID
	stadiumLoaded = 0

	def episode_restart(self, bullet_client):
		self._p = bullet_client
		Scene.episode_restart(self, bullet_client)
		if self.stadiumLoaded == 0:
			self.stadiumLoaded = 1

			filename = os.path.join(os.path.dirname(__file__), "..", "assets", "scenes", "plane_stadium.sdf")
			self.ground_plane_mjcf=self._p.loadSDF(filename)

			for i in self.ground_plane_mjcf:
				self._p.changeDynamics(i,-1,lateralFriction=0.8, restitution=0.5)
				self._p.changeVisualShape(i,-1,rgbaColor=[1,1,1,0.8])
				self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION,1)
