# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation

"""Utility functions to calculate Laikago's pose and motor angles."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import attr

# LAIKAGO_DEFAULT_ABDUCTION_ANGLE = 0
# LAIKAGO_DEFAULT_HIP_ANGLE = 0.67
# LAIKAGO_DEFAULT_KNEE_ANGLE = -1.25

LAIKAGO_DEFAULT_ABDUCTION_ANGLE = 0
LAIKAGO_DEFAULT_HIP_ANGLE = 0.9
LAIKAGO_DEFAULT_KNEE_ANGLE = -1.8

@attr.s
class LaikagoPose(object):
  """Default pose of the Laikago.

    Leg order:
    0 -> Front Right.
    1 -> Front Left.
    2 -> Rear Right.
    3 -> Rear Left.
  """
  abduction_angle_0 = attr.ib(type=float, default=0)
  hip_angle_0 = attr.ib(type=float, default=0)
  knee_angle_0 = attr.ib(type=float, default=0)
  abduction_angle_1 = attr.ib(type=float, default=0)
  hip_angle_1 = attr.ib(type=float, default=0)
  knee_angle_1 = attr.ib(type=float, default=0)
  abduction_angle_2 = attr.ib(type=float, default=0)
  hip_angle_2 = attr.ib(type=float, default=0)
  knee_angle_2 = attr.ib(type=float, default=0)
  abduction_angle_3 = attr.ib(type=float, default=0)
  hip_angle_3 = attr.ib(type=float, default=0)
  knee_angle_3 = attr.ib(type=float, default=0)
