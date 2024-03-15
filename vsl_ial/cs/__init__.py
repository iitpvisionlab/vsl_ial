from __future__ import annotations
import importlib
from typing import Optional, Literal
import numpy.typing as npt
import numpy as np
from .. import FArray


class whitepoints_cie1931:
    # The "standard" 2 degree observer (CIE 1931)
    # Sourced from http://www.easyrgb.com/index.php?X=MATH&H=15
    A = np.asfarray([1.09850, 1.0, 0.35585])
    C = np.asfarray([0.98074, 1.0, 1.18232])
    D50 = np.asfarray([0.96422, 1.0, 0.82521])
    D55 = np.asfarray([0.95682, 1.0, 0.92149])
    D65 = np.asfarray([0.95047, 1.0, 1.08883])
    D75 = np.asfarray([0.94972, 1.0, 1.22638])
    F2 = np.asfarray([0.99186, 1.0, 0.67393])
    F7 = np.asfarray([0.95041, 1.0, 1.08747])
    F11 = np.asfarray([1.00962, 1.0, 0.64350])


class whitepoints_cie1964:
    # The "supplementary" 10 degree observer (CIE 1964)
    # Sourced from http://www.easyrgb.com/index.php?X=MATH&H=15
    A = np.asfarray([1.11144, 1.0, 0.35200])
    C = np.asfarray([0.97285, 1.0, 1.16145])
    D50 = np.asfarray([0.96720, 1.0, 0.81427])
    D55 = np.asfarray([0.95799, 1.0, 0.90926])
    D65 = np.asfarray([0.94811, 1.0, 1.07304])
    D75 = np.asfarray([0.94416, 1.0, 1.20641])
    F2 = np.asfarray([1.03279, 1.0, 0.69027])
    F7 = np.asfarray([0.95792, 1.0, 1.07686])
    F11 = np.asfarray([1.03863, 1.0, 0.65607])


D65 = whitepoints_cie1931.D65


class CS:
    def __init__(self, illuminant_xyz: Optional[FArray] = None):
        self._illuminant_xyz = illuminant_xyz

    def convert(self, src: CS, color: FArray) -> FArray:
        try:
            from_func = getattr(self, f"_from_{src.__class__.__name__}")
        except AttributeError:
            try:
                to_func = getattr(src, f"_to_{self.__class__.__name__}")
            except AttributeError:
                type_self = self.__class__

                # same color - do nothing
                if type(src) is type_self:
                    return color

                # try going through 'XYZ'
                e = None
                if type_self.__name__ != "XYZ":
                    try:
                        from .xyz import XYZ

                        xyz = XYZ()
                        color_xyz = convert(src, xyz, color)
                        return convert(xyz, self, color_xyz)
                    except Exception as ee:
                        e = ee

                # give up
                raise NotImplementedError(
                    f"{src.__class__.__name__} is not yet able to convert to "
                    f"{self.__class__.__name__}"
                ) from e
            else:
                return to_func(self, color)
        else:
            return from_func(src, color)

    def distance(self, a: FArray, b: FArray, ord=None) -> float:
        return np.linalg.norm(a - b, ord=ord, axis=a.ndim - 1)


def convert(
    src: CS, dst: CS, color: npt.ArrayLike, dtype: npt.DTypeLike = None
) -> FArray:
    color = np.asfarray(color, dtype)
    return dst.convert(src=src, color=color)


def distance(
    cs: CS,
    color1: npt.ArrayLike,
    color2: npt.ArrayLike,
    dtype: npt.DTypeLike = None,
    name: (
        Literal[""]
        | Literal["ciede2000"]
        | Literal["cie76"]
        | Literal["cie94"]
        | Literal["cbLCH"]
        | Literal["cbLAB"]
        | Literal["HyAB"]
        | Literal["HyCH"]
    ) = "",
    **kwargs: float,
) -> float:
    color1 = np.asfarray(color1, dtype)
    color2 = np.asfarray(color2, dtype)
    try:
        method = getattr(cs, "distance" + (f"_{name}" if name else ""))
    except AttributeError:
        raise ValueError(
            f"distance '{name}' does not make sense for {cs.__class__.__name__}"
        )
    return method(color1, color2, **kwargs)


available_cs = {
    "CAM02": "cam",
    "CAM02LCD": "cam",
    "CAM02SCD": "cam",
    "CAM02UCS": "cam",
    "CAM16": "cam",
    "CAM16LCD": "cam",
    "CAM16SCD": "cam",
    "CAM16UCS": "cam",
    "CIELAB": "cielab",
    "CIELUV": "cieluv",
    "CIExyY": "ciexyy",
    "ICaCb": "icacb",
    "ICtCp": "ictcp",
    "JzAzBz": "jzazbz",
    "linRGB": "linrgb",
    "LMS": "lms",
    "Oklab": "oklab",
    "Opponent": "opponent",
    "ProLab": "prolab",
    "sRGB": "srgb",
    "XYZ": "xyz",
}

__all__ = (
    "D65",
    "convert",
    "distance",
    "whitepoints_cie1931",
    "whitepoints_cie1964",
) + tuple(available_cs)


def __getattr__(cs_name: str) -> CS:
    try:
        module_name = available_cs[cs_name]
    except Exception:
        raise AttributeError(
            f"module {__name__!r} has no attribute {cs_name!r}"
        )
    module = importlib.import_module(f".{module_name}", __name__)
    return getattr(module, cs_name)
