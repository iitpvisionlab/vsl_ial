from __future__ import annotations
from typing import Literal
from ._base import StrictModel
from ..datasets.distance.constant_hue import (
    load_ebner_fairchild,
    load_hung_berns,
    load_xiao_averages,
)
from .cs import CS
from .. import FArray
from ..cs.xyz import XYZ
import numpy as np
from rich.console import Console
from rich.table import Table


class DatasetConstantHue(StrictModel):
    pass


class DatasetEbnerFairchild(DatasetConstantHue):
    name: Literal["ebner_fairchild"] = "ebner_fairchild"

    def load(self):
        return load_ebner_fairchild()


class DatasetHungBerns(DatasetConstantHue):
    name: Literal["hung_berns"] = "hung_berns"

    def load(self):
        return load_hung_berns()


class DatasetXiao(DatasetConstantHue):
    name: Literal["xiao"] = "xiao"

    def load(self):
        return load_xiao_averages()


class Config(StrictModel):
    cs: list[CS]
    datasets: list[DatasetEbnerFairchild | DatasetHungBerns | DatasetXiao]


def _metric(angles_rad: FArray) -> tuple[float, float]:
    tau = 2.0 * np.pi
    assert np.all(angles_rad >= 0.0)
    assert np.all(angles_rad < tau)

    sin_mean = np.mean(np.sin(angles_rad))
    cos_mean = np.mean(np.cos(angles_rad))
    mean_angle_rad = np.arctan2(sin_mean, cos_mean) % tau

    diff = (angles_rad - mean_angle_rad) % tau
    diff[diff > np.pi] -= tau

    return np.rad2deg(np.std(diff, ddof=1))


def main(config: Config, console: Console | None = None):
    console = console or Console()

    table = Table(title=f"UCS Hue Nonlinearity (SD units, lower is better).")
    table.add_column("")
    datasets = [dataset.load() for dataset in config.datasets]
    for dataset in datasets:
        table.add_column(dataset.name)

    for cs_model in config.cs:
        row: list[str] = [cs_model.name]
        for dataset in datasets:
            std_list: list[float] = []
            cs = cs_model.create_for(dataset)
            tau = 2.0 * np.pi
            for same in dataset.items:
                color = cs.from_XYZ(XYZ(), same.colors)
                a, b = cs.chromaticity(color)
                h = (np.arctan2(a, b) + tau) % tau
                std_list.append(_metric(h))
            row.append(f"{np.mean(std_list):.04g}")
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    main()
