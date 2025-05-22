from __future__ import annotations
from . import CS, FArray
import numpy as np


M_LMS = np.array(
    (
        (0.674207838, 0.149284160, 0.070941080),
        (0.382799340, 0.739628340, 0.174768000),
        (-0.047570458, 0.083327300, 0.670970020),
    ),
    dtype=np.float64,
)

M_LMS_INV = np.array(
    (
        (
            +1.661373055774069e00,
            -3.250758740427037e-01,
            -9.098281098284756e-02,
        ),
        (
            -9.145230923250668e-01,
            +1.571847038366936e00,
            -3.127282905230740e-01,
        ),
        (
            +2.313620767186147e-01,
            -2.182538318672940e-01,
            +1.522766561305260e00,
        ),
    ),
    dtype=np.float64,
)

M_IAZ = np.array(
    (
        (0.5, 3.524000, 0.199076),
        (0.5, -4.066708, 1.096799),
        (0.0, 0.542708, -1.295875),
    ),
    dtype=np.float64,
)

M_IAZ_INV = np.array(
    (
        (1.0, 1.0, 1.0),
        (1.386050432715393e-1, -1.386050432715393e-1, -9.601924202631895e-2),
        (5.804731615611869e-2, -5.804731615611891e-2, -8.118918960560390e-1),
    ),
    dtype=np.float64,
)


class JzAzBz(CS):
    #
    # https://observablehq.com/@jrus/jzazbz
    #
    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        def pqInv(X: FArray) -> FArray:
            XX = (X * 1e-4) ** 0.1593017578125
            return (
                (0.8359375 + 18.8515625 * XX) / (1 + 18.6875 * XX)
            ) ** 134.034375

        LMSp = pqInv(np.tensordot(color, M_LMS, axes=1))
        Iaz = np.tensordot(LMSp, M_IAZ, axes=1)  # ЙАААААЗЬ
        Iz = Iaz[..., 0]
        Iaz[..., 0] = (0.44 * Iz) / (1 - 0.56 * Iz) - 1.6295499532821566e-11
        return Iaz

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        Jz = color[..., 0] + 1.6295499532821566e-11
        color[..., 0] = Jz / (0.44 + 0.56 * Jz)

        def pqInv(X: FArray) -> FArray:
            XX = X**7.460772656268214e-03
            return (
                1e4
                * ((0.8359375 - XX) / (18.6875 * XX - 18.8515625))
                ** 6.277394636015326
            )

        LMS = pqInv(np.tensordot(color, M_IAZ_INV, axes=1))

        return np.tensordot(LMS, M_LMS_INV, axes=1)
