from __future__ import annotations
from typing import Callable, TypeAlias, Any
from pathlib import Path
from collections.abc import Sequence

import json5
from .config import Config
from vsl_ial.cs.pcs23 import PCS23UCS
from vsl_ial import FArray
from scipy.optimize import minimize
from .dataset import Dataset
import numpy as np
from vsl_ial.stress import Ord

Float = np.floating[Any]

LossFunction: TypeAlias = Callable[
    [Sequence[FArray], Sequence[FArray], Ord], Float
]


def train(
    model_cls: type[PCS23UCS],
    loaded_datasets: list[list[Dataset]],
    loss_function: LossFunction,
) -> None:

    def evaluate(x: list[float]) -> float:
        model = model_cls(x[:39], H=x[39:])
        ref: list[FArray] = []
        exp: list[FArray] = []

        for datasets in loaded_datasets:
            for dataset in datasets:
                model_coordinates = model.from_XYZ(None, dataset.xyz)
                a_colors = np.empty((len(dataset.pairs), 3), dtype=np.float64)
                b_colors = np.empty_like(a_colors)
                for idx, (a_idx, b_idx) in enumerate(dataset.pairs):
                    a_colors[idx] = model_coordinates[a_idx]
                    b_colors[idx] = model_coordinates[b_idx]
                exp_distance = np.linalg.norm(
                    a_colors - b_colors, axis=1, ord=2
                )
                exp.append(exp_distance)
                ref.append(dataset.dv)
        loss = loss_function(ref, exp)
        print("loss function", loss)
        return loss

    # x0 = [0.2] * (39 + 8)
    # x0 = np.random.rand(39 + 8)
    x0 = np.asarray(
        [
            5.65822087e00,
            1.75511708e02,
            1.16367546e03,
            -2.00084521e-03,
            -5.05773364e02,
            -1.36521727e05,
            -5.89062148e02,
            -2.64563249e04,
            -9.57581167e03,
            2.35986631e02,
            6.55043332e02,
            4.46391903e-01,
            1.58078875e03,
            1.24616075e04,
            -1.34185422e00,
            1.78546715e05,
            2.19355970e02,
            -9.81213831e04,
            -5.46287245e02,
            1.88035463e00,
            3.18578603e00,
            -5.09896100e00,
            8.85655496e02,
            7.94394362e-01,
            1.76772051e01,
            -1.73650355e02,
            -3.29927222e01,
            -8.72088578e03,
            2.35704637e03,
            3.77036132e01,
            -2.88774290e01,
            -1.03504007e03,
            9.84241706e01,
            3.32752406e00,
            -4.28197266e02,
            2.31725935e04,
            -6.81158416e01,
            9.27573345e02,
            -9.31335222e01,
            3.79644431e02,
            -9.22254559e01,
            -8.04988702e01,
            -8.18506374e01,
            3.74275903e02,
            -1.15349017e02,
            -4.35768649e01,
            1.68770308e02,
        ]
    )
    # np.random.shuffle(x0)
    for i in range(5):
        res = minimize(
            fun=evaluate,
            method="Nelder-Mead",
            x0=x0,
            tol=1e-4,
            options={"maxiter": 2000},
        )
        x0 = res.x
    breakpoint()
    print("minimization result", res)


def get_xyz():
    with open(
        "/home/senyai/projects/vsl_ial/vsl_ial/datasets/ciexyz31_1.csv"
    ) as f:
        lines: list[tuple[int, float, float, float]] = []
        for line in f:
            wl, x, y, z = line.split(",")
            lines.append((int(wl), float(x), float(y), float(z)))
    return lines


def main():
    #     wl_xyz = get_xyz()
    #     xyz = np.asarray([row[1:] for row in wl_xyz])
    #     xs, ys, zs = xyz.T

    #     x = [ 5.65822087e+00,  1.75511708e+02,  1.16367546e+03, -2.00084521e-03,
    # -5.05773364e+02, -1.36521727e+05, -5.89062148e+02, -2.64563249e+04,
    # -9.57581167e+03,  2.35986631e+02,  6.55043332e+02,  4.46391903e-01,
    #  1.58078875e+03,  1.24616075e+04, -1.34185422e+00,  1.78546715e+05,
    #  2.19355970e+02, -9.81213831e+04, -5.46287245e+02,  1.88035463e+00,
    #  3.18578603e+00, -5.09896100e+00,  8.85655496e+02,  7.94394362e-01,
    #  1.76772051e+01, -1.73650355e+02, -3.29927222e+01, -8.72088578e+03,
    #  2.35704637e+03,  3.77036132e+01, -2.88774290e+01, -1.03504007e+03,
    #  9.84241706e+01,  3.32752406e+00, -4.28197266e+02,  2.31725935e+04,
    # -6.81158416e+01,  9.27573345e+02, -9.31335222e+01,  3.79644431e+02,
    # -9.22254559e+01, -8.04988702e+01, -8.18506374e+01,  3.74275903e+02,
    # -1.15349017e+02, -4.35768649e+01,  1.68770308e+02]
    #     model = PCS23UCS()

    #     ucs = model.from_XYZ(None, xyz)
    #     xs, ys, zs = ucs.T

    #     import matplotlib.pyplot as plt

    #     fig = plt.figure()
    #     ax = fig.add_subplot(projection='3d')
    #     ax.scatter(xs, ys, zs)
    #     plt.show()

    #     return
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = Config(**json5.loads(args.config.read_text()))
    loaded_datasets = [dataset.load() for dataset in config.datasets]
    loss = config.loss.load()
    model = PCS23UCS
    print(
        f"loaded_datasets = {[[dataset.name for dataset in datasets] for datasets in loaded_datasets]}"
    )
    print(f"model = {model}")
    print(f"loss = {loss}")
    train(model, loaded_datasets, loss)
