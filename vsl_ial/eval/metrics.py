from __future__ import annotations

from typing import Literal, Annotated
from pydantic import Field
from ._base import StrictModel
from .. import FArray


class StressLoss(StrictModel):
    name: Literal["stress"]

    def load(self):
        from vsl_ial.stress import stress
        import numpy as np

        def stress_(seq1: list[FArray], seq2: list[FArray]):
            return stress(np.hstack(seq1), np.hstack(seq2))

        return stress_


class GroupSTRESSLoss(StrictModel):
    name: Literal["group_stress"]

    def load(self):
        from vsl_ial.stress import group_stress

        return group_stress


Metrics = Annotated[
    StressLoss | GroupSTRESSLoss,
    Field(..., discriminator="name"),
]
