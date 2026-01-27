from __future__ import annotations
from typing import Callable, TypeAlias, Any
from time import perf_counter
from pathlib import Path
from collections.abc import Sequence
from datetime import timedelta

import json5
import numpy as np
from scipy.optimize import minimize

from vsl_ial import FArray
from vsl_ial.cs.pcs23 import PCS23UCS, CS
from vsl_ial.datasets.sensitivities import load as load_sensitivity
from vsl_ial.cs.xyz import XYZ
from vsl_ial.cs.ciexyy import CIExyY

from ..eval._base import StrictModel
from ..eval.dataset import DatasetConfig, WeightedDataset
from ..eval.metrics import Metrics


class Config(StrictModel):
    datasets: list[DatasetConfig]
    loss: Metrics


Float = np.floating[Any]
F32Array = np.ndarray[Any, np.dtype(np.float32)]

LossFunction: TypeAlias = Callable[[Sequence[FArray], Sequence[FArray]], Float]


def _point_inside(x: float, y: float, poly: list[tuple[float, float]]):
    """
    ToDo: this is a very slow function.
          Can be optimized 100x after this module is working
    """
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class MonotonicityLoss:
    def __init__(self) -> None:
        full_xy_grid = self._create_grid(
            np.linspace(0, 1, 30, dtype=np.float32),
            np.linspace(0, 1, 30, dtype=np.float32),
        )
        self.Y = Y = np.linspace(0.02, 1.0, 49)
        self._sensitivity_xyz = sensitivity_xyz = np.asarray(
            load_sensitivity("cie-1931-2")["xyz"], dtype=np.float64
        ).T
        spectral_locus_xy = (
            CIExyY().from_XYZ(XYZ(), sensitivity_xyz)[:, :2].tolist()
        )
        xy_points = np.asarray(
            [
                pt
                for pt in full_xy_grid
                if _point_inside(*pt.tolist(), spectral_locus_xy)
            ],
            dtype=np.float32,
        )
        self.xy_points = xy_points
        xy_points_repeated = xy_points[None].repeat(Y.size, 0)
        n_points = len(xy_points)
        xyY = np.dstack(
            (xy_points_repeated, Y.repeat(n_points).reshape(-1, n_points, 1))
        )
        self.XYZ_diff = CIExyY().to_XYZ(XYZ(), xyY)
        self.xyY_diff = xyY

    def __call__(self, cs: CS) -> float:
        """
        Formula 10 of PCS23-UCS
        """
        res = cs.from_XYZ(XYZ(), self.XYZ_diff)
        minimum_diff = 0.02
        L_plus = res[..., 0]
        θ = float(
            np.mean(
                np.maximum(
                    0,
                    -np.diff(L_plus, axis=0) / np.diff(self.Y)[..., None]
                    + minimum_diff,
                )
            )
        )
        return θ

    @staticmethod
    def _create_grid(x: F32Array, y: F32Array) -> F32Array:
        return np.dstack(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)


def train(
    model_cls: type[PCS23UCS],
    loaded_datasets: list[WeightedDataset],
    loss_function: LossFunction,
) -> None:
    monotonicity_loss = MonotonicityLoss()

    def evaluate(x: list[float]) -> float:

        opt_model = model_cls(
            F_LA_or_D=0.0, illuminant_xyz=None, V=x[:39], H=x[39:]
        )

        stress = 0.0
        for loaded_dataset in loaded_datasets:
            ref: list[FArray] = []
            exp: list[FArray] = []
            for dataset in loaded_dataset.datasets:
                assert dataset.F is not None, dataset
                model = model_cls(
                    F_LA_or_D=(dataset.F, dataset.L_A),
                    illuminant_xyz=dataset.illuminant,
                    V=x[:39],
                    H=x[39:],
                )

                model_coordinates = model.from_XYZ(XYZ(), dataset.xyz)
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
            stress += loss_function(ref, exp) * loaded_dataset.weight
        monotonicity = monotonicity_loss(opt_model)
        loss = stress + 0.1 * monotonicity
        print(f"{stress=}, {monotonicity=}, {loss=}")
        return loss

    x0 = np.random.rand(39 + 8)
    start = perf_counter()
    for i in range(60):
        res = minimize(
            fun=evaluate,
            method="Nelder-Mead",
            x0=x0,
            tol=1e-2,
            options={"maxiter": 150, "maxfev": 150},
        )
        print(
            f"step {i+1}: {res.x.tolist()} t={timedelta(seconds=perf_counter() - start)}"
        )
        x0 = res.x
    print("minimization result", res)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--update-schema", action="store_true")
    args = parser.parse_args()
    if args.update_schema:
        schema_path = Path(__file__).with_name("schema.json")
        schema_path.write_text(
            json5.dumps(
                Config.model_json_schema(), ensure_ascii=False, quote_keys=True
            )
        )
        return
    config = Config(**json5.loads(args.config.read_text()))
    loaded_datasets = [dataset.load() for dataset in config.datasets]
    loss = config.loss.load()
    model = PCS23UCS
    for loaded_dataset in loaded_datasets:
        print(f"* {loaded_dataset.name}")
        if len(loaded_dataset.datasets) != 1:
            for subset in loaded_dataset.datasets:
                print(f" - {subset.name}")

    print(f"model = {model}")
    print(f"loss = {loss}")
    train(model, loaded_datasets, loss)
