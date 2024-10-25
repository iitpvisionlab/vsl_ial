from __future__ import annotations
from . import CS, FArray
import numpy as np
from .hls import HLS


class HSV(CS):
    """
    https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    def _from_sRGB(self, src: CS, color: FArray) -> FArray:
        h, l, s = HLS()._from_sRGB(src, color).T
        v = l + s * np.minimum(l, 1.0 - l)
        s = 2 * (1.0 - np.divide(l, v, where=v != 0, out=np.ones_like(v)))
        return np.dstack((h, s, v)).reshape(color.shape)

    def _to_sRGB(self, src: CS, color: FArray) -> FArray:
        h, s, v = color.T
        l = v * (1.0 - s / 2.0)
        minl = np.minimum(l, 1.0 - l)
        s = np.divide(
            (v - l), minl, where=minl != 0.0, out=np.zeros_like(minl)
        )
        return HLS()._to_sRGB(self, np.dstack((h, l, s)).reshape(color.shape))
