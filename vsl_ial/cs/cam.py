"""
Comprehensive color solutions: CAM16, CAT16, and CAM16-UCS
"""

from __future__ import annotations
from abc import abstractmethod
from . import CS, convert, FArray, Ord
from typing import Tuple, Union, NamedTuple, ClassVar
import numpy as np


M_16 = np.array(
    (
        (+0.401288, +0.650173, -0.051461),
        (-0.250268, +1.204414, +0.045854),
        (-0.002079, +0.048952, +0.953127),
    ),
    dtype=np.float64,
)
""" 1. Li et al. (2017) Comprehensive colour solutions: CAM16, CAT16 and CAM16-UCS
2. Green & Habib (2019) Chromatic adaptation in colour management
"""

M_16_INV = np.array(
    (
        (+1.86206786, -1.01125463, +0.14918677),
        (+0.38752654, +0.62144744, -0.00897398),
        (-0.01584150, -0.03412294, +1.04996444),
    ),
    dtype=np.float64,
)
"""1. Li et al. (2017) Comprehensive colour solutions: CAM16, CAT16 and CAM16-UCS"""


M_CAT02 = np.array(
    (
        (+0.7328, +0.4296, -0.1624),
        (-0.7036, +1.6975, +0.0061),
        (+0.0030, +0.0136, +0.9834),
    ),
    dtype=np.float64,
)


M_CAT02_INV = np.array(
    (
        (+1.096124, -0.278869, 0.182745),
        (+0.454369, +0.473533, 0.072098),
        (-0.009628, -0.005698, 1.015326),
    ),
    dtype=np.float64,
)

M_HPE = np.array(
    (
        (+0.38971, +0.68898, -0.07868),
        (-0.22981, +1.18340, +0.04641),
        (+0.00000, +0.00000, +1.00000),
    ),
    dtype=np.float64,
)


class Surround(NamedTuple):
    """
    *CIECAM02* colour appearance model induction factors.

    See TABLE A1 Surround parameters
    """

    F: float
    c: float
    N_c: float


Average = Surround(1, 0.69, 1)
Dim = Surround(0.9, 0.59, 0.9)
Dark = Surround(0.8, 0.525, 0.8)


class CAMCommon(CS):
    RGB_a_coefs = (
        np.array(
            ((460, +451, +288), (460, -891, -261), (460, -220, -6300)),
            dtype=np.float64,
        )
        / 1403
    )
    M: FArray

    def __init__(
        self,
        illuminant_xyz: FArray,
        L_A: float,
        Y_b: float,
        surround: Surround,
    ):
        """
        L_A: Luminance of test adapting field :math:`cd/m^2`
        Y_b: Luminous factor of background in percent. Typical value is 20.
        """
        super().__init__(illuminant_xyz)
        assert 0.525 <= surround.c <= 0.69
        self.surround = surround
        self.n = Y_b * 0.01 / illuminant_xyz[..., 1]
        self.z = 1.48 + np.sqrt(self.n)

        self.N_bb = 0.725 * self.n**-0.2
        assert -2.0 < self.N_bb < 2.0, self.N_bb  # sanity check

        self.N_cb = self.N_bb

        k = 1.0 / (5.0 * L_A + 1.0)
        k4 = 0.0 if np.isinf(L_A) else k**4
        self.F_L = k4 * L_A + 0.1 * (1.0 - k4) ** 2 * np.cbrt(5.0 * L_A)
        self.F_L_root4 = self.F_L**0.25

        self.D_RGB = self.calc_d_rgb(
            M=self.M,
            illuminant_src=illuminant_xyz,
            illuminant_dst=np.array((1.0, 1.0, 1.0)),
            F_LA_or_D=(surround.F, L_A),
        )
        _, self.A_w = self._postadaptation_cone_response(illuminant_xyz)

    @abstractmethod
    def _response(self, color: FArray) -> FArray:
        pass

    @abstractmethod
    def _response_inv(self, color: FArray) -> FArray:
        pass

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        """
        Returns JMh coordinates
        """
        rgb_a_, A = self._postadaptation_cone_response(color)
        return self._calculate_jmh(rgb_a_, A).reshape(color.shape)

    def to_XYZ(self, dst: CS, color: FArray) -> FArray:
        """Input: J; M; h"""
        J, M, h = color.T
        cos_h = np.cos(h)
        sin_h = np.sin(h)
        J_root = np.sqrt(J)
        alpha = (M / self.F_L_root4) / J_root
        t = (alpha * 100.0 * ((1.64 - 0.29**self.n) ** -0.73)) ** (10.0 / 9.0)
        e_t = 0.25 * (np.cos(h + 2) + 3.8)
        z = 1.48 + np.sqrt(self.n)
        A = self.A_w * J_root ** (2 / self.surround.c / z)
        p_1 = 5e4 / 13 * self.surround.N_c * self.N_cb * e_t
        p_2 = A / self.N_bb
        r = (
            23
            * (p_2 + 0.305)
            * t
            / (23 * p_1 + t * (11 * cos_h + 108 * sin_h))
        )
        a = r * cos_h
        b = r * sin_h

        RGB_a = np.array((p_2, a, b)).T @ self.RGB_a_coefs.T
        RGB_a_abs = np.abs(RGB_a)
        constant = 1.0 / self.F_L * 27.13 ** (1.0 / 0.42)
        RGB_c = (
            np.sign(RGB_a)
            * constant
            * (RGB_a_abs / (400.0 - RGB_a_abs)) ** (1.0 / 0.42)
        )
        return self._response_inv(RGB_c)

    def _postadaptation_cone_response(self, color: FArray):
        rgb_c = self._response(color)

        x = (self.F_L * np.abs(rgb_c)) ** 0.42
        rgb_a_ = np.sign(rgb_c) * 400.0 * x / (x + 27.13)

        # step 6
        A = self.N_bb * (rgb_a_ @ [2.0, 1.0, 0.05])
        return rgb_a_, A

    def _calculate_jmh(self, rgb_a_: FArray, A: float):
        # step 5 -- same as step 4 in CAM16
        a = rgb_a_ @ [1.0, -12 / 11, 1 / 11]
        b = (rgb_a_ @ [1.0, 1.0, -2.0]) / 9
        h = np.arctan2(b, a)

        # step 6 -- same as step 5 in CAM16
        # radians(20.14) == 0.35150931135165797
        # radians(360.0) == 6.283185307179586
        h_shift = np.where(h < 0.35150931135165797, h + 6.283185307179586, h)
        e_t = (np.cos(h_shift + 2) + 3.8) / 4

        # step 7
        # we get A from step 3

        # step 8 -- same as step 7 in CAM16
        J_root = (A / self.A_w) ** (self.surround.c * self.z * 0.5)
        J = np.square(J_root)

        # step 9
        # pass, we don't need Q for UCS

        # step 10 -- same as step 9 in CAM16
        t_denum = rgb_a_ @ [1.0, 1.0, 21 / 20] + 0.305
        t = (
            (5e4 * self.surround.N_c * self.N_cb / 13)
            * e_t
            * np.hypot(a, b)
            / t_denum
        )

        C = t**0.9 * J**0.5 * (1.64 - 0.29**self.n) ** 0.73

        M = C * self.F_L_root4 * 0.01

        return np.dstack((J, M, h_shift))

    @staticmethod
    def calc_d_rgb(
        M: FArray,
        illuminant_src: FArray,
        illuminant_dst: FArray,
        F_LA_or_D: Union[Tuple[float, float], float],
    ) -> FArray:
        rgb_w = M @ illuminant_src
        rgb_wr = M @ illuminant_dst
        Y_w: float = illuminant_src[1]
        Y_wr: float = illuminant_dst[1]

        # Step 0
        if isinstance(F_LA_or_D, tuple):
            #
            # Calculate all values/parameters which are independent
            # of the input sample
            #
            F, L_A = F_LA_or_D
            D = F * (1.0 - np.exp((-L_A - 42.0) / 92.0) / 3.6)
            # If D is greater than one or less than zero,
            # set it to one or zero, respectively.
            D = np.clip(D, 0.0, 1.0)
        else:
            D = F_LA_or_D
            assert 0.0 <= D <= 1.0, D
        D_RGB = D * (Y_w * rgb_wr) / (Y_wr * rgb_w) + 1 - D
        return D_RGB


class _CATBase:
    M: ClassVar[FArray]
    M_INV: ClassVar[FArray]

    def __init__(
        self,
        illuminant_src: FArray,
        illuminant_dst: FArray,
        F_LA_or_D: tuple[float, float] | float,
        exact: bool = True,
    ):
        D_RGB = CAMCommon.calc_d_rgb(
            self.M, illuminant_src, illuminant_dst, F_LA_or_D
        )
        self._M = (
            (np.linalg.solve(self.M, (self.M.T * D_RGB).T)).T
            if exact
            else (self.M_INV @ (self.M.T * D_RGB).T).T
        )

    def __call__(self, xyz: FArray):
        return xyz @ self._M


class CAT02(_CATBase):
    M = M_CAT02
    M_INV = M_CAT02_INV


class CAT16(_CATBase):
    M = M_16
    M_INV = M_16_INV


class CAM16(CAMCommon):
    M = M_16

    def _response(self, color: FArray) -> FArray:
        return self.D_RGB * np.tensordot(color, self.M.T, axes=1)

    def _response_inv(self, color: FArray) -> FArray:
        XYZ = (color / self.D_RGB) @ M_16_INV.T
        return XYZ


class CAM02(CAMCommon):
    M = M_CAT02

    def _response(self, color: FArray) -> FArray:
        new_m = M_HPE @ M_CAT02_INV @ (self.M.T * self.D_RGB).T
        return np.tensordot(color, new_m.T, axes=1)

    def _response_inv(self, color: FArray) -> FArray:
        Minv = M_CAT02_INV @ ((self.M).T / self.D_RGB).T
        Minv = Minv @ np.linalg.inv(M_HPE)
        XYZ = np.tensordot(color, Minv.T, axes=1)
        return XYZ


class _CAMBase(CS):
    c1 = 0.007 * 100.0
    cls: ClassVar[type[CAMCommon]]
    c2: ClassVar[float]
    k: ClassVar[float]

    def __init__(
        self,
        illuminant_xyz: FArray,
        L_A: float,
        Y_b: float,
        surround: Surround,
    ):
        self._cam = self.cls(illuminant_xyz, L_A, Y_b, surround)

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        J, M, h = convert(src, self._cam, color).T
        M_ = np.log(1 + self.c2 * M) / self.c2
        J_ = 1.7 * J / (1 + self.c1 * J)
        return np.array([J_, M_ * np.cos(h), M_ * np.sin(h)]).T

    def to_XYZ(self, dst: CS, color: FArray):
        J_, a, b = color.reshape(-1, 3).T
        J = -J_ / (self.c1 * (J_ - 1) - 1)
        h = np.mod(np.arctan2(b, a), np.pi * 2.0)
        M_ = np.hypot(a, b)
        M = (np.exp(M_ * self.c2) - 1.0) / self.c2
        return convert(
            self._cam, dst, np.dstack([J, M, h]).reshape(color.shape)
        )

    def distance(self, a: FArray, b: FArray, ord: Ord = None) -> float:
        diff = np.divide(a - b, [self.k, 1.0, 1.0])
        return np.linalg.norm(diff, ord=ord, axis=a.ndim - 1)


class CAM16UCS(_CAMBase):
    c2 = 0.0228 * 100.0
    cls = CAM16
    k = 1.0


class CAM16LCD(_CAMBase):
    c2 = 0.0053 * 100.0
    cls = CAM16
    k = 0.77


class CAM16SCD(_CAMBase):
    c2 = 0.0363 * 100.0
    cls = CAM16
    k = 1.24


class CAM02UCS(_CAMBase):
    c2 = 0.0228 * 100.0
    cls = CAM02
    k = 1.0


class CAM02LCD(_CAMBase):
    c2 = 0.0053 * 100.0
    cls = CAM02
    k = 0.77


class CAM02SCD(_CAMBase):
    c2 = 0.0363 * 100.0
    cls = CAM02
    k = 1.24
