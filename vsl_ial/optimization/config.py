from __future__ import annotations

from typing import Literal, Annotated
from pydantic import Field
from ._base import StrictModel
from .dataset import DatasetConfig


class StressLoss(StrictModel):
    name: Literal["stress"]

    def load(self):
        from vsl_ial.stress import stress

        def stress_(seq1, seq2):
            assert len(seq1) == len(seq2) == 1, (len(seq1), len(seq2))
            return stress(seq1[0], seq2[0])

        return stress_


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
