"""
Usage example:

statistics = image_metric(
    convert(sRGB(), XYZ(), A),
    convert(sRGB(), XYZ(), B),
    filter_color_space=Opponent(),
    spacial_filters=(create_gauss_kernel_2d(1.0), create_gauss_kernel_2d(1.0), create_gauss_kernel_2d(1.0)),
    metric_color_space=CIELAB(illuminant_xyz=D65))
"""

from __future__ import annotations
import numpy as np
from scipy.signal import convolve
from .cs.xyz import XYZ
from .cs import convert, CS, FArray
from typing import NamedTuple


class Statistics(NamedTuple):
    metric_image: FArray
    mean: float
    max: float
    std: float

    @classmethod
    def create(cls, metric_image: FArray):
        return cls(
            metric_image=metric_image,
            mean=np.mean(metric_image),
            max=np.max(metric_image),
            std=np.std(metric_image),
        )


def create_gauss_kernel_2d(sigma: float, weight: float = 1.0):
    # https://en.wikipedia.org/wiki/Gaussian_blur
    size = np.ceil(sigma * 3.5)
    x = np.arange(-size, size + 1)[:, np.newaxis]
    y = x.T
    gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gauss *= weight / gauss.sum()
    return gauss


def image_metric(
    A_xyz: FArray,
    B_xyz: FArray,
    filter_color_space: CS,
    spacial_filters: tuple[FArray, FArray, FArray],
    metric_color_space: CS,
) -> Statistics:
    assert A_xyz.shape == B_xyz.shape, (A_xyz.shape, B_xyz.shape)
    assert A_xyz.ndim == 3, A_xyz.ndim
    assert len(spacial_filters) == 3
    for filter in spacial_filters:
        for sz in filter.shape:
            assert sz % 2 == 1, f"filter size must be odd (not {sz})"
    A = convert(XYZ(), filter_color_space, A_xyz)
    B = convert(XYZ(), filter_color_space, B_xyz)

    A_convolved = np.zeros_like(A)
    B_convolved = np.zeros_like(A)
    for img_src, img_dst in ((A, A_convolved), (B, B_convolved)):
        for ch_idx, kernel in enumerate(spacial_filters):
            w, h = kernel.shape[0:2]
            assert w == h
            sz = w // 2  # can be easily changed to support non square kernels
            ch_in_padded = np.pad(img_src[:, :, ch_idx], sz, "reflect")
            img_dst[:, :, ch_idx] = convolve(ch_in_padded, kernel, "valid")
    A_metric_cs = convert(filter_color_space, metric_color_space, A_convolved)
    B_metric_cs = convert(filter_color_space, metric_color_space, B_convolved)
    metric = np.linalg.norm(A_metric_cs - B_metric_cs, axis=2)

    return Statistics.create(metric)


def image_metric_itp(
    A_xyz: FArray, B_xyz: FArray, p: float = 0.75, gamma_value: float = 1.0
) -> Statistics:
    from .image_appearance_model.icam06 import iCAM06_HDR

    A_ipt = iCAM06_HDR(A_xyz, p=p, gamma_value=gamma_value).ipt
    B_ipt = iCAM06_HDR(B_xyz, p=p, gamma_value=gamma_value).ipt
    metric = np.linalg.norm(A_ipt - B_ipt, axis=2)
    return Statistics.create(metric)
