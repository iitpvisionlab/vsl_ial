from __future__ import annotations
from . import CS, FArray
import numpy as np
import warnings


LIN_RGB_MATRIX = np.asarray(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    ),
    dtype=np.float64,
).T


class linRGB(CS):
    def __init__(self, illuminant_xyz: FArray | None = None):
        assert illuminant_xyz is None
        super().__init__()

    def _from_XYZ(self, src: CS, color: FArray) -> FArray:
        assert src.__class__.__name__ == "XYZ"
        return color @ LIN_RGB_MATRIX

    def _from_sRGB(self, src: CS, color: FArray) -> FArray:
        if color.min() < 0 or color.max() > 1:
            warnings.warn(
                f"sRGB range should be in [0, 1] not [{color.min()}, {color.max()}]"
            )

        thres = 12.92 * 0.0031308
        a = 0.055

        if color.dtype == np.uint8:
            lut = np.empty((256,), dtype=np.float32)
            for lut_idx in range(256):
                value = lut_idx / 255
                low = value <= thres
                if low:
                    lut[lut_idx] = value / 12.92
                else:
                    lut[lut_idx] = ((value + a) / (1 + a)) ** 2.4
            color = lut[color]
        else:
            mask = color > 0.04045
            color = color.copy()
            color[mask] = np.power((color[mask] + 0.055) / 1.055, 2.4)
            color[~mask] /= 12.92
        return color

    def _to_XYZ(self, src: CS, color: FArray) -> FArray:
        return color @ np.linalg.inv(LIN_RGB_MATRIX)
