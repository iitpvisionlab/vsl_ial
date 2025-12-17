from __future__ import annotations
from . import CS, FArray
import numpy as np


class HLS(CS):
    """
    https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    def from_sRGB(self, src: CS, color: FArray) -> FArray:
        maxc = np.max(color, axis=-1)
        minc = np.min(color, axis=-1)
        sumc = maxc + minc
        rangec = maxc - minc
        l = sumc * 0.5
        has_hue = minc != maxc
        saturation = np.divide(
            rangec,
            np.where(l <= 0.5, sumc, 2.0 - sumc),
            where=has_hue,
            out=np.zeros_like(sumc),
        )
        c = np.divide(
            (maxc[..., None] - color),
            rangec[..., None],
            where=has_hue[..., None],
            out=np.zeros_like(color),
        )
        r = color[..., 0]
        g = color[..., 1]
        rc = c[..., 0]
        gc = c[..., 1]
        bc = c[..., 2]
        hue = np.where(
            r == maxc,
            bc - gc,
            np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
        )
        hue = (hue / 6.0) % 1.0
        return np.dstack((hue, l, saturation)).reshape(color.shape)

    def to_sRGB(self, src: CS, color: FArray) -> FArray:
        h, l, s = color.reshape(-1, 3).T
        m2 = np.where(l <= 0.5, l * (1.0 + s), l + s - (l * s))
        m1 = 2.0 * l - m2

        def _v(m1: FArray, m2: FArray, hue: FArray) -> FArray:
            hue = hue % 1.0
            return np.select(
                (hue < 1 / 6, hue < 0.5, hue < 2 / 3),
                (
                    (m1 + (m2 - m1) * hue * 6.0),
                    m2,
                    m1 + (m2 - m1) * (2 / 3 - hue) * 6.0,
                ),
                m1,
            )

        return np.dstack(
            (
                _v(m1, m2, h + 1 / 3),
                _v(m1, m2, h),
                _v(m1, m2, h - 1 / 3),
            )
        ).reshape(color.shape)
