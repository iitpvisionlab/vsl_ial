from __future__ import annotations
from . import CS, FArray
import numpy as np


class ProLab(CS):
    q = np.array((0.7554, 3.8666, 1.6739), dtype=np.float64)

    Q = (
        np.array(
            (
                (75.54, 617.72, 48.34),
                (486.66, -595.45, 194.94),
                (167.39, -22.27, -243.28),
            ),
            dtype=np.float64,
        )
        / 100.0
    )

    Q_inv = np.array(
        (
            (0.13706328211735358, 0.13706328211735355, 0.13706328211735358),
            (0.1387382031383206, -0.024315485429340655, 0.008083459429919239),
            (0.08160688511070953, 0.09653291949249931, -0.3174818967768846),
        ),
        dtype=np.float64,
    )

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        color_ = color / self._illuminant_xyz
        return (
            (np.tensordot(color_, self.Q, axes=1)).T
            / (np.tensordot(color_, self.q, axes=1).T + 1.0)
        ).T

    def to_XYZ(self, dst: CS, color: FArray) -> FArray:
        y2 = np.tensordot(color, self.Q_inv, axes=1)
        xyz = y2.T / (1.0 - np.tensordot(y2, self.q, axes=1)).T
        return xyz.T * self._illuminant_xyz
