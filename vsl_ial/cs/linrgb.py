from __future__ import annotations
from . import CS, FArray
import numpy as np
import warnings


LIN_RGB_MATRIX = np.array(
    (
        (3.2404542, -0.9692660, 0.0556434),
        (-1.5371385, 1.8760108, -0.2040259),
        (-0.4985314, 0.0415560, 1.0572252),
    ),
    dtype=np.float64,
)

LIN_RGB_MATRIX_INV = np.array(
    (
        (0.4124564, 0.2126729, 0.0193339),
        (0.3575761, 0.7151522, 0.1191920),
        (0.1804375, 0.0721750, 0.9503041),
    ),
    dtype=np.float64,
)


class linRGB(CS):
    def __init__(self, illuminant_xyz: FArray | None = None):
        assert illuminant_xyz is None
        super().__init__()

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        return color @ LIN_RGB_MATRIX

    def from_sRGB(self, src: CS, color: FArray) -> FArray:
        if color.min() < 0 or color.max() > 1:
            # this warning is usually harmless
            warnings.warn(
                f"sRGB range should be in [0, 1] not "
                f"[{color.min()}, {color.max()}]"
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

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        return color @ LIN_RGB_MATRIX_INV
