from __future__ import annotations
import numpy as np
from . import CS, FArray, whitepoints_cie1964
from .ciexyy import CIExyY
from .cam import CAT16


def _rotate_y(xyz: FArray, beta: float) -> FArray:
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    rotation_matrix = np.array(
        [
            [cos_beta, 0.0, -sin_beta],
            [0.0, 1.0, 0.0],
            [sin_beta, 0.0, cos_beta],
        ],
        dtype=np.float64,
    )
    return xyz @ rotation_matrix


def _rotate_z(xyz: FArray, gamma: float) -> FArray:
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    rotation_matrix = np.array(
        [
            [cos_gamma, sin_gamma, 0.0],
            [-sin_gamma, cos_gamma, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return xyz @ rotation_matrix


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

    There are no differences in values compared to original article:
    * input XYZ range is [0, 1]
    * output range is [-100, 100].

    Details:
    * `illuminant_xyz` can be `None`. When it is `None`, chromatic adaptation
      step is disabled
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
        F_LA_or_D: tuple[float, float] | float | None,
        illuminant_xyz: FArray | None = whitepoints_cie1964.D65,
        V: tuple[float, ...] = DEFAULT_V,
        H: tuple[float, ...] = DEFAULT_H,
    ):
        super().__init__(illuminant_xyz)
        assert len(V) == 39, f"{len(V)}"
        assert len(H) == 8, f"{len(H)}"
        self._V = np.vstack((V[:20], np.hstack((np.array(V[20:]), 1.0))))
        self._H = np.array(
            (
                H[:3],
                H[3:6],
                np.hstack((np.asarray(H[6:8]), 1)),
            )
        )
        E = np.array((0.01, 0.01, 0.01))
        self._cat16 = (
            None
            if illuminant_xyz is None
            else CAT16(
                illuminant_src=illuminant_xyz,
                illuminant_dst=E,
                F_LA_or_D=F_LA_or_D,
            )
        )
        self._white_point = self._convert(E)

    @staticmethod
    def _Tc(h: FArray) -> FArray:
        return h / h[-1]

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        if self._cat16 is not None:
            color = self._cat16(color)
        raw_pcones = self._convert(color)
        k = 1.0 / np.linalg.norm(self._white_point)
        scaled_pcones = k * raw_pcones
        scaled_wp = k * self._white_point
        a_angle = np.arctan2(scaled_wp[2], scaled_wp[0])
        pcones_rot_y = _rotate_y(scaled_pcones, a_angle)
        wp_rot_y = _rotate_y(scaled_wp, a_angle)
        b_angle = -np.arctan2(wp_rot_y[1], wp_rot_y[0])
        pcones = _rotate_z(pcones_rot_y, b_angle)
        return pcones

    def _convert(self, color: FArray) -> FArray:
        xyY = CIExyY().from_XYZ(self, color)
        x, y, Y = xyY.reshape(-1, 3).T
        yY = np.divide(y, Y, where=Y != 0, out=np.zeros_like(Y))
        x_sq = np.square(x)
        y_sq = np.square(y)
        yY_sq = np.square(yY)

        # NOTE! the order is not like in formula 6, but as in values table
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
