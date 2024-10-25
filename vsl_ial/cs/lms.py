from __future__ import annotations
from . import CS, FArray
import numpy as np


class LMS(CS):
    """
    1. http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
    2. Fairchild, M. (2005). Color appearance in image displays, p. 177
    """

    VON_KRIES_M = np.asarray(
        (
            (0.40024, 0.70760, -0.08081),
            (-0.2263, 1.16532, 0.04570),
            (0.0, 0.0, 0.9182200),
        ),
        dtype=np.float64,
    )
    VON_KRIES_M_INV = np.asarray(
        (
            (1.8599364, -1.1293816, 0.2198974),
            (0.3611914, 0.6388125, -0.0000064),
            (0.0, 0.0, 1.0890636),
        ),
        dtype=np.float64,
    )

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        return color @ self.VON_KRIES_M.T

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        return color @ self.VON_KRIES_M_INV.T
