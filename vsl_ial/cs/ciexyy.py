from __future__ import annotations
from . import CS, FArray
import numpy as np


class CIExyY(CS):
    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        sum_ = color.reshape(-1, 3).sum(axis=-1).T
        X, Y, _Z = color.reshape(-1, 3).T
        return np.dstack((X / sum_, Y / sum_, Y)).reshape(color.shape)

    def to_XYZ(self, dst: CS, color: FArray) -> FArray:
        x, y, Y = color.reshape(-1, 3).T
        Y_div_y = Y / y
        return np.dstack((x * Y_div_y, Y, (1.0 - x - y) * Y_div_y)).reshape(
            color.shape
        )
