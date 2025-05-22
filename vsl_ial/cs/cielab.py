"""
https://en.wikipedia.org/wiki/Lab_color_space
"""

from __future__ import annotations
from typing import NamedTuple
import numpy as np

from vsl_ial import FArray
from . import CS, FArray


class _CIEDE2000(NamedTuple):
    # intermediate values needed for different algorithms
    ΔL: FArray
    Sl: FArray
    ΔC: FArray
    Sc: FArray
    ΔH: FArray
    Sh: FArray
    C_mean: FArray
    h_mean: FArray


def f(t: FArray):
    delta = 6 / 29
    return np.where(
        t > delta**3, 1.16 * np.cbrt(t) - 0.16, 0.01 * t / (delta / 2) ** 3
    )


def finv(t: FArray):
    delta = 6 / 29
    t_ = t / 1.16
    return np.where(t_ > 0.02 / 0.29, (t_ + 4 / 29) ** 3, 3 * delta**2 * t_)


class CIELAB(CS):
    A = np.array(
        (
            (0.0, 125 / 29, 0.0),
            (1.0, -125 / 29, 50 / 29),
            (0.0, 0.0, -50 / 29),
        ),
        dtype=np.float64,
    )
    Ainv = np.array(
        (
            (1.0, 1.0, 1.0),
            (29 / 125, 0.0, 0.0),
            (0.0, 0.0, -116 / 200),
        ),
        dtype=np.float64,
    )

    _illuminant_xyz: FArray

    def __init__(self, illuminant_xyz: FArray):
        assert illuminant_xyz is not None
        super().__init__(illuminant_xyz)

    def from_XYZ(self, src: CS, color: FArray):
        return np.tensordot(f(color / self._illuminant_xyz), self.A, axes=1)

    def to_XYZ(self, dst: CS, color: FArray):
        return (
            finv(np.tensordot(color, self.Ainv, axes=1)) * self._illuminant_xyz
        )

    distance_cie76 = CS.distance

    def distance_cie94(
        self,
        a: FArray,
        b: FArray,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ) -> float:
        """https://en.wikipedia.org/wiki/Color_difference"""
        L1, a1, b1 = a.T
        L2, a2, b2 = b.T
        ΔL = L1 - L2
        C_ab_1 = np.hypot(a1, b1)
        C_ab_2 = np.hypot(a2, b2)
        ΔC_ab = C_ab_1 - C_ab_2
        Δa = a1 - a2
        Δb = b1 - b2
        ΔH = np.sqrt(
            (np.square(Δa) + np.square(Δb) - np.square(ΔC_ab)).clip(min=0.0)
        )
        SL = 1.0
        # C_ab_x = np.sqrt(C_ab_1 * C_ab_2)
        SC = 1.0 + 0.045 * C_ab_1
        SH = 1.0 + 0.015 * C_ab_1

        return np.sqrt(
            np.square(ΔL / (kL * SL))
            + np.square(ΔC_ab / (kC * SC))
            + np.square(ΔH / (kH * SH))
        )

    @staticmethod
    def _ciede2000(a: FArray, b: FArray) -> _CIEDE2000:
        L1, a1, b1 = a.T
        L2, a2, b2 = b.T

        C_ab_1 = np.hypot(a1, b1)
        C_ab_2 = np.hypot(a2, b2)

        C_ab_mean7 = (0.5 * (C_ab_1 + C_ab_2)) ** 7.0

        G = 1.0 + 0.5 * (1.0 - np.sqrt(C_ab_mean7 / (C_ab_mean7 + 25.0**7.0)))

        a1_scaled = a1 * G
        a2_scaled = a2 * G

        C_scaled_1 = np.hypot(a1_scaled, b1)
        C_scaled_2 = np.hypot(a2_scaled, b2)

        tau = np.pi * 2.0
        h_1 = np.arctan2(b1, a1_scaled) % tau
        h_2 = np.arctan2(b2, a2_scaled) % tau

        ΔL = L2 - L1
        ΔC = C_scaled_2 - C_scaled_1
        Δh = (h_2 - h_1 + np.pi) % tau - np.pi  # ensure [-np.pi, np.pi) range
        ΔH = 2.0 * np.sqrt(C_scaled_1 * C_scaled_2) * np.sin(Δh * 0.5)

        L_mean = (L1 + L2) * 0.5
        C_mean = (C_scaled_1 + C_scaled_2) * 0.5

        h_mean = np.where(
            np.fabs(h_1 - h_2) <= np.pi,
            (h_1 + h_2) * 0.5,
            ((h_1 + h_2 + tau) * 0.5),
        )

        T = (
            1.0
            - 0.17 * np.cos(h_mean - np.radians(30.0))
            + 0.24 * np.cos(h_mean * 2.0)
            + 0.32 * np.cos(h_mean * 3.0 + np.radians(6.0))
            - 0.20 * np.cos(h_mean * 4.0 - np.radians(63.0))
        )

        tmp = np.square(L_mean - 50.0)
        Sl = 1.0 + (0.015 * tmp) / np.sqrt(20.0 + tmp)
        Sc = 1.0 + 0.045 * C_mean
        Sh = 1.0 + 0.015 * C_mean * T

        return _CIEDE2000(ΔL, Sl, ΔC, Sc, ΔH, Sh, C_mean, h_mean)

    @classmethod
    def distance_ciede2000(
        cls,
        a: FArray,
        b: FArray,
        ord=None,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ) -> float:
        """
        A Top Down Description ofS-CIELAB and CIEDE2000Garrett M. Johnson,* Mark D. Fairchild
        """
        res = cls._ciede2000(a, b)

        ΔΘ = np.radians(30.0) * np.exp(
            -np.square((res.h_mean - np.radians(275.0)) / np.radians(25.0))
        )
        C_mean7 = res.C_mean**7.0
        RC = 2.0 * np.sqrt(C_mean7 / (C_mean7 + 25.0**7.0))
        RT = -np.sin(2.0 * ΔΘ) * RC

        ΔC_tmp = res.ΔC / (kC * res.Sc)
        ΔH_tmp = res.ΔH / (kH * res.Sh)
        return np.sqrt(
            np.square(res.ΔL / (kL * res.Sl))
            + np.square(ΔC_tmp)
            + np.square(ΔH_tmp)
            + RT * ΔC_tmp * ΔH_tmp
        )

    @classmethod
    def distance_cbLCH(
        cls,
        a: FArray,
        b: FArray,
        ord=None,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ):
        """City Block in (L*C*H*) aka CD3"""
        res = cls._ciede2000(a, b)
        return (
            np.abs(res.ΔL / (kL * res.Sl))
            + np.abs(res.ΔC / (kC * res.Sc))
            + np.abs(res.ΔH / ((kH * res.Sh)))
        )

    @staticmethod
    def distance_cbLAB(a: FArray, b: FArray):
        """City Block in (L*a*b*) aka CD2"""
        return np.abs(a - b).sum(axis=min(a.ndim - 1, 1))

    @staticmethod
    def distance_HyAB(a: FArray, b: FArray):
        """aka CD1"""
        ΔL, Δa, Δb = (a - b).T
        return np.abs(ΔL) + np.hypot(Δa, Δb)

    @classmethod
    def distance_HyCH(
        cls,
        a: FArray,
        b: FArray,
        kL: float = 1.0,
        kC: float = 1.0,
        kH: float = 1.0,
    ):
        """aka CD4"""
        res = cls._ciede2000(a, b)
        return np.abs(res.ΔL / (kL * res.Sl)) + np.hypot(
            res.ΔC / (kC * res.Sc), res.ΔH / (kH * res.Sh)
        )
