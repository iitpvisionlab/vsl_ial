from __future__ import annotations

from typing import Literal, Annotated
from pydantic import Field
from ._base import StrictModel
from .dataset import DatasetConfig


class StressLoss(StrictModel):
    name: Literal["stress"]

    def load(self):
        from vsl_ial.stress import stress

        return stress


class GroupSTRESSLoss(StrictModel):
    name: Literal["group_stress"]

    def load(self):
        from vsl_ial.stress import group_stress

        return group_stress


Loss = Annotated[
    StressLoss | GroupSTRESSLoss,
    Field(..., discriminator="name"),
]


class Config(StrictModel):
    datasets: list[DatasetConfig]
    loss: Loss
