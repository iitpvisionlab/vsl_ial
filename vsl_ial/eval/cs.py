from __future__ import annotations

from typing import Literal, Annotated, TypeAlias, TYPE_CHECKING
from pydantic import Field
from ._base import StrictModel
from ..datasets.distance import DistanceDataset
from .. import FArray

if TYPE_CHECKING:
    from ..cs.cam import (
        CAM16UCS as CAM16UCS_,
        CAM16LCD as CAM16LCD_,
        CAM16SCD as CAM16SCD_,
        CAM02UCS as CAM02UCS_,
        CAM02LCD as CAM02LCD_,
        CAM02SCD as CAM02SCD_,
    )

    AnyCAM: TypeAlias = type[
        CAM16UCS_ | CAM16LCD_ | CAM16SCD_ | CAM02UCS_ | CAM02LCD_ | CAM02SCD_
    ]


class CAMBase(StrictModel):
    def create_for(self, dataset: DistanceDataset):
        from ..cs.cam import Surround

        return self.model(
            illuminant_xyz=dataset.illuminant,
            L_A=dataset.L_A,
            Y_b=dataset.Y_b,
            surround=Surround(
                F=dataset.F,
                c=dataset.c,
                N_c=dataset.Nc,
            ),
        )

    @property
    def model(self) -> AnyCAM:
        raise NotImplementedError()


class CAM16SCD(CAMBase):
    name: Literal["CAM16-SCD"]

    @property
    def model(self):
        from ..cs.cam import CAM16SCD

        return CAM16SCD


class Debug(StrictModel):
    name: Literal["CAM16SCD-color"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs import CS
        import colour

        surround = colour.appearance.InductionFactors_CIECAM16(
            F=dataset.F, c=dataset.c, N_c=dataset.Nc
        )

        class Debug(CS):
            def from_XYZ(self, src: CS, color: FArray) -> FArray:
                cam16_scd = colour.XYZ_to_CAM16SCD(
                    color,
                    XYZ_w=dataset.illuminant,
                    L_A=dataset.L_A,
                    Y_b=dataset.Y_b,
                    surround=surround,
                )
                return cam16_scd

        return Debug()


class CAM16UCS(CAMBase):
    name: Literal["CAM16-UCS"]

    @property
    def model(self):
        from ..cs.cam import CAM16UCS

        return CAM16UCS


class CAM16LCD(CAMBase):
    name: Literal["CAM16-LCD"]

    @property
    def model(self):
        from ..cs.cam import CAM16LCD

        return CAM16LCD


class CAM02SCD(CAMBase):
    name: Literal["CAM02-SCD"]

    @property
    def model(self):
        from ..cs.cam import CAM02SCD

        return CAM02SCD


class CAM02UCS(CAMBase):
    name: Literal["CAM02-UCS"]

    @property
    def model(self):
        from ..cs.cam import CAM02UCS

        return CAM02UCS


class CAM02LCD(CAMBase):
    name: Literal["CAM02-LCD"]

    @property
    def model(self):
        from ..cs.cam import CAM02LCD

        return CAM02LCD


class PCS23UCS(StrictModel):
    name: Literal["PCS23-UCS"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.pcs23 import PCS23UCS

        return PCS23UCS(
            F_LA_or_D=(dataset.F, dataset.L_A),
            illuminant_xyz=dataset.illuminant,
        )


class CIELAB(StrictModel):
    name: Literal["CIELAB"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.cielab import CIELAB

        return CIELAB(illuminant_xyz=dataset.illuminant)


class ProLab(StrictModel):
    name: Literal["ProLab"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.prolab import ProLab

        return ProLab(illuminant_xyz=dataset.illuminant)


class Oklab(StrictModel):
    name: Literal["Oklab"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.oklab import Oklab

        return Oklab(illuminant_xyz=dataset.illuminant)


class JzAzBz(StrictModel):
    name: Literal["JzAzBz"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.jzazbz import JzAzBz

        return JzAzBz()


class ICaCb(StrictModel):
    name: Literal["ICaCb"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.icacb import ICaCb

        return ICaCb()


class ICtCp(StrictModel):
    name: Literal["ICtCp"]

    def create_for(self, dataset: DistanceDataset):
        from ..cs.ictcp import ICtCp

        return ICtCp()


CS = Annotated[
    CAM02LCD
    | CAM02SCD
    | CAM02UCS
    | CAM16LCD
    | CAM16SCD
    | CAM16UCS
    | CIELAB
    | Debug  # Debug is here, but not int the list by default
    | ICaCb
    | ICtCp
    | JzAzBz
    | Oklab
    | PCS23UCS
    | ProLab,
    Field(..., discriminator="name"),
]
