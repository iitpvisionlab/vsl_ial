from __future__ import annotations
from . import CS, FArray
import numpy as np


XYZ_TO_LMS = np.array(
    (
        (0.8189330101, 0.0329845436, 0.0482003018),
        (0.3618667424, 0.9293118715, 0.2643662691),
        (-0.1288597137, 0.0361456387, 0.6338517070),
    ),
    dtype=np.float64,
)

LMS_TO_LAB = np.array(
    (
        (0.2104542553, 1.9779984951, 0.0259040371),
        (0.7936177850, -2.4285922050, 0.7827717662),
        (-0.0040720468, 0.4505937099, -0.8086757660),
    ),
    dtype=np.float64,
)

XYZ_TO_LMS_INV = np.array(
    (
        (1.2270138511035211, -0.04058017842328059, -0.07638128450570689),
        (-0.5577999806518223, 1.11225686961683, -0.42148197841801266),
        (0.2812561489664678, -0.0716766786656012, 1.5861632204407947),
    ),
    dtype=np.float64,
)

LMS_TO_LAB_INV = np.array(
    (
        (0.9999999984505199, 1.0000000088817607, 1.0000000546724108),
        (0.3963377921737679, -0.10556134232365635, -0.08948418209496575),
        (0.2158037580607588, -0.06385417477170591, -1.2914855378640917),
    ),
    dtype=np.float64,
)


class Oklab(CS):
    """
    Oklab color space,
    https://bottosson.github.io/posts/oklab/
    """

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        lms = np.tensordot(color, XYZ_TO_LMS, axes=1)
        lms_cubic_root = np.cbrt(lms)
        return np.tensordot(lms_cubic_root, LMS_TO_LAB, axes=1)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        lab = np.tensordot(color, LMS_TO_LAB_INV, axes=1)
        lab_cube = lab**3.0
        return np.tensordot(lab_cube, XYZ_TO_LMS_INV, axes=1)
