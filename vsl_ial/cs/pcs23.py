from __future__ import annotations
from . import CS, FArray
from .ciexyy import CIExyY
import numpy as np


class PCS23UCS(CS):
    """
    Implementation of

    O. A. Basova, V. P. Bozhkova, I. A. Konovalenko, A. P. Sarycheva,
    M. K. Chobanu, V. A. Timofeev and D. P. Nikolaev,
    “Uniform Color Space with Advanced Hue Linearity: PCS23-UCS,”
    CCIW 2024, 15193 ed., Raimondo Schettini, Alain Trémeau, Shoji Tominaga,
    Simone Bianco, Marco Buzzelli, Ed., Cham, Switzerland,
    Springer Nature Group, Lecture Notes in Computer Science (LNCS),
    2024, vol. 15193, ISSN 0302-9743, ISBN 978-3-031-72844-0, no 15193,
    pp. 36-50, 2025, DOI: 10.1007/978-3-031-72845-7_3.
    """

    DEFAULT_V = (
        -10.359368263560516,
        0.865216921303402,
        -0.17852501058411485,
        72.35212968566003,
        58.373464583098894,
        0.17759576581185665,
        -82.23939162009282,
        5.374236418250991,
        -0.10898351765548173,
        -34.92698266388017,
        -6.896458430832865,
        0.0,
        -28.230578892499715,
        0.019890445320401733,
        -0.07519161732425433,
        -7.942790452007076,
        18.295674422273677,
        -0.3165550687255816,
        0.6255241814142221,
        0.9133947986237523,
        0.8314034024748209,
        9.619118314317834,
        3.5426045972473243,
        41.051688944461176,
        -15.803935117841657,
        0.6212170182027368,
        -111.84944828034054,
        -15.444993521186333,
        -2.6545034288489378,
        24.606934759082442,
        110.34330362029283,
        0.12642025855546973,
        0.7272676826476265,
        0.35928591448377256,
        -1.7126837452286745,
        116.23513491673151,
        20.870649205517182,
        71.75689264938393,
        -145.50060558025496,
    )

    DEFAULT_H = (
        1.7409668308820407,
        -6.825272149685501,
        1.5444010948816445,
        5.479730974967994,
        -1.2274248802562973,
        -1.15959074596261,
        -0.4363169336227266,
        -0.4745662634207841,
    )

    def __init__(
        self,
        V: tuple[float, ...] = DEFAULT_V,
        H: tuple[float, ...] = DEFAULT_H,
        illuminant_xyz: FArray | None = None,
    ):
        super().__init__(illuminant_xyz)
        assert len(V) == 39, f"{len(V)}"
        assert len(H) == 8, f"{len(H)}"
        self._V = np.vstack((V[:20], np.hstack((np.array(V[20:]), 1.0))))
        self._H = np.asarray(
            (
                H[:3],
                H[3:6],
                np.hstack((np.asarray(H[6:8]), 1)),
            )
        )

    @staticmethod
    def _Tc(h: FArray) -> FArray:
        return h / h[-1]

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        xyY = CIExyY().from_XYZ(self, color)
        x, y, Y = xyY.T
        x = x.reshape(-1)
        y = y.reshape(-1)
        Y = Y.reshape(-1)
        Y *= 100.0
        yY = np.divide(y, Y, where=Y != 0, out=np.zeros_like(Y))
        x_sq = np.square(x)
        y_sq = np.square(y)
        yY_sq = np.square(yY)

        # WARNING! the order is not like on formula 6, but as in values table
        thp33 = np.vstack(
            (
                x,
                y,
                yY,
                x_sq,
                y_sq,
                yY_sq,
                x * y,
                y * yY,
                x * yY,
                x**3,
                y**3,
                yY**3,
                x * y_sq,
                x * yY_sq,
                y * yY_sq,
                yY * y_sq,
                y * x_sq,
                yY * x_sq,
                x * y * yY,
                np.ones_like(x),
            )
        )
        L_plus = self._Tc(self._V @ thp33)[0]

        xy = np.vstack((x, y))
        thxy = np.vstack((xy, np.ones_like(x)))
        x_plus_y_plus = self._Tc(self._H @ thxy)[:-1]

        return np.vstack((L_plus, x_plus_y_plus * L_plus)).T.reshape(
            color.shape
        )

    def to_XYZ(self, dst: CS, color: FArray) -> FArray:
        """
        the PCS23-UCS transformation is continuous and invertible within the
        visible color gamut if two conditions are met.
        """
        raise NotImplementedError()
