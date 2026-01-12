from __future__ import annotations
from typing import Any, Sequence, Literal, TYPE_CHECKING
from . import FArray
import numpy as np
from numpy.linalg import norm

if TYPE_CHECKING:
    Float = np.floating[Any]
Ord = Literal[1, 2]


def calc_k(a: FArray, b: FArray, ord: Ord) -> Float:
    if ord == 2:
        return (a * b).sum() / np.square(a).sum()
    elif ord == 1:
        b_div_a = b / a
        sorted_indices = b_div_a.argsort()
        csum = a[sorted_indices].cumsum()
        idx = np.searchsorted(csum, csum[-1] * 0.5, "left")
        try:
            idx = sorted_indices[idx]
        except IndexError:
            raise ValueError("all values must be above 0.0")
        return b_div_a[idx]
    else:
        raise ValueError(f"invalid ord={ord}")


def stress(a: FArray, b: FArray, ord: Ord = 2) -> Float:
    assert a.ndim == b.ndim == 1, (a.shape, b.shape)
    return norm(calc_k(a, b, ord) * a - b, ord) / norm(b, ord)


def mean_stress(
    seq_a: Sequence[FArray], seq_b: Sequence[FArray], ord: Ord = 2
) -> Float:
    _sizes, stresses = zip(
        *((a.shape[0], stress(a, b, ord=ord)) for a, b in zip(seq_a, seq_b))
    )
    return np.mean(stresses)
    # return (sizes * stresses).sum() / n.sum()


def group_stress(
    seq_a: Sequence[FArray],
    seq_b: Sequence[FArray],
    ord: Ord = 2,
) -> Float:
    assert len(seq_a) == len(seq_b)
    vals = np.concatenate(
        [calc_k(a, b, ord) * a - b for a, b in zip(seq_a, seq_b)]
    )
    return norm(np.hstack(vals), ord) / norm(np.hstack(seq_b), ord)


def weighted_group_stress(
    seq_a: Sequence[FArray],
    seq_b: Sequence[FArray],
    weights: Sequence[FArray | Float | float],
    ord: Ord = 2,
) -> Float:
    assert len(seq_a) == len(seq_b)
    assert len(seq_a) == len(weights)
    if ord == 2:
        weights = [np.sqrt(w) for w in weights]
    vals = np.concatenate(
        [
            (calc_k(a, b, ord) * a - b) * w
            for a, b, w in zip(seq_a, seq_b, weights)
        ]
    )
    l = [v * w for v, w in zip(seq_b, weights)]
    return norm(np.hstack(vals), ord) / norm(np.hstack(l), ord)


def cv(e: list[FArray], v: list[FArray]) -> float:
    """
    CV metric
    """
    e_, v_ = np.hstack(e), np.hstack(v)
    f = (e_ * v_).sum() / np.square(v).sum()
    return 100.0 * np.sqrt(
        np.mean(np.square(e_ - f * v_) / np.square(np.mean(e_)))
    )


def pf3(e: list[FArray], v: list[FArray]) -> float:
    """
    PF/3 metric
    """
    e_, v_ = np.hstack(e), np.hstack(v)
    if not np.all(v_ > 0):
        raise ValueError()
    e_div_v = e_ / v_
    F = np.sqrt(e_div_v.sum() / (v_ / e_).sum())
    f = (e_ * v_).sum() / np.square(v_).sum()
    Vab = np.sqrt(np.mean(np.square(e_ - F * v_) / e_ * F * v_))
    CV_ = np.sqrt(np.mean(np.square(e_ - f * v_) / np.square(np.mean(e_))))

    log10_e_div_v = np.log10(e_div_v)
    γ = 10.0 ** np.sqrt(
        np.mean(np.square(log10_e_div_v - np.mean(log10_e_div_v)))
    )
    return (100.0 * (γ - 1.0) + Vab + CV_) / 3.0
