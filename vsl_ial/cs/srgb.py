from __future__ import annotations
from . import CS, FArray
import numpy as np
import warnings
from .linrgb import linRGB
from . import convert
from .xyz import XYZ


class sRGB(CS):
    def from_linRGB(self, src: CS, color: FArray) -> FArray:
        assert src.__class__.__name__ == "linRGB"
        thres = 0.0031308
        a = 0.055

        if color.min() < 0 or color.max() > 1:
            warnings.warn(
                "When converting for linRGB to sRGB, values should be in "
                f"range [0, 1] not [{color.min()} {color.max()}]"
            )

        color_clipped = np.clip(color, 0.0, 1.0)
        color_clipped_f = color_clipped.reshape(-1)

        for y in range(0, color_clipped.size, 4096):
            fragment = color_clipped_f[y : y + 4096]
            low = fragment <= thres

            fragment[low] *= 12.92
            fragment[~low] = (1 + a) * fragment[~low] ** (1 / 2.4) - a

        return color_clipped

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        linrgb = linRGB(self._illuminant_xyz)
        color = linrgb.from_XYZ(src, color)
        return self.from_linRGB(linrgb, color)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        color_linRGB = convert(sRGB(), linRGB(), color=color)
        return convert(
            linRGB(illuminant_xyz=self._illuminant_xyz),
            XYZ(),
            color=color_linRGB,
        )
