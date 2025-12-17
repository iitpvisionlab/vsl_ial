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
        denominator = (color @ [1.0, 15.0, 3.0])[..., None]

        XY = color[..., 0:2]
        XYs = [[4.0, 9.0]] * XY
        uv = np.divide(
            XYs,
            denominator,
            out=np.zeros_like(XYs),
            where=denominator != 0,
        )

        l = f(color[..., 1] / wp[1])[..., None]

        uv2 = 13.0 * l * (uv - self.uv_wp)
        return np.column_stack((l.reshape(-1, 1), uv2.reshape(-1, 2))).reshape(
            color.shape
        )

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        l, uv = color[..., 0].reshape(-1, 1), color[..., 1:3].reshape(-1, 2)
        puv = uv / (13.0 * l) + self.uv_wp
        y = finv(l)
        up, vp = puv.reshape(-1, 1, 2).T
        d = 4.0 * vp.T
        ret = np.column_stack(
            (
                y * 9.0 * up.T / d,
                y,
                y * (12.0 - 3.0 * up.T - 20.0 * vp.T) / d,
            )
        ).reshape(color.shape)
        return ret
