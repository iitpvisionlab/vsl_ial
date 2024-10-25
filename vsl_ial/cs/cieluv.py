from __future__ import annotations
from . import CS, FArray
import numpy as np
from .cielab import f, finv


class CIELUV(CS):
    """
    CIE 1976 L* u* v* (https://en.wikipedia.org/wiki/CIELUV)

    https://observablehq.com/@mbostock/luv-and-hcl
    """

    def __init__(self, illuminant_xyz: FArray | None = None):
        assert illuminant_xyz is not None
        super().__init__(illuminant_xyz)
        self.uv_wp = (
            illuminant_xyz[0:2]
            * [4.0, 9.0]
            / (illuminant_xyz @ [1.0, 15.0, 3.0])
        )

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        wp = self._illuminant_xyz
        denominator = (color @ [1.0, 15.0, 3.0]).T

        XY = color[..., 0:2]
        uv = np.divide(
            [4.0, 9.0] * XY,
            denominator,
            out=np.zeros_like(XY),
            where=denominator != 0,
        )

        l = f(color[..., 1] / wp[1])

        uv = 13.0 * l.T * (uv - self.uv_wp)
        return np.dstack((l.T, *uv.T)).reshape(color.shape)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        l, uv = color[..., 0], color[..., 1:3]
        puv = uv / (13.0 * l.T) + self.uv_wp
        y = finv(l.T)
        up, vp = puv.T
        d = 4 * vp
        ret = np.dstack(
            (y * 9.0 * up / d, y, y * (12.0 - 3.0 * up - 20.0 * vp) / d)
        ).reshape(color.shape)
        return ret
