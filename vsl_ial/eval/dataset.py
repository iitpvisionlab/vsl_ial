from __future__ import annotations
from typing import Literal, Annotated, ClassVar

from pydantic import Field
from ._base import StrictModel
from typing import NamedTuple
from ..datasets.distance import (
    DistanceDataset,
    load_bfd_p,
    load_leeds,
    load_munsell,
    load_rit_dupont,
    load_witt,
    MunsellGroup,
    MunsellContains,
)


class WeightedDataset(NamedTuple):
    datasets: list[DistanceDataset]
    weight: float
    name: str


class BaseDataset(StrictModel):
    weight: float = 1.0

    def load(self) -> WeightedDataset:
        raise NotImplementedError(self)


class DatasetBfd_p(BaseDataset):
    name: Literal["bfd_p"] = "bfd_p"
    subsets: list[Literal["C", "D65", "M"]] = ["C", "D65", "M"]

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_bfd_p(subsets=[i.lower() for i in self.subsets]),
            weight=self.weight,
            name=self.name,
        )

    @property
    def display_name(self):
        return f"BFD-P {', '.join(self.subsets)}"


class DatasetFairchild_chen(BaseDataset):
    name: Literal["fairchild_chen"]


class DatasetHung_berns(BaseDataset):
    name: Literal["hung_berns"]


class DatasetIlluminants(BaseDataset):
    name: Literal["illuminants"]


class DatasetLeeds(BaseDataset):
    name: Literal["leeds"] = "leeds"
    display_name: ClassVar[str] = "Leeds"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_leeds(), weight=self.weight, name=self.name
        )


class DatasetLuo_rigg(BaseDataset):
    name: Literal["luo_rigg"]
    display_name: ClassVar[str] = "Luo Rigg"


class DatasetMacadam_1942(BaseDataset):
    name: Literal["macadam_1942"]
    display_name: ClassVar[str] = "MacAdam 1942"


class DatasetMacadam_1974(BaseDataset):
    name: Literal["macadam_1974"]
    display_name: ClassVar[str] = "MacAdam 1974"


class DatasetMunsell(BaseDataset):
    name: Literal["munsell"]
    version: Literal["2.0", "3.0", "3.1.0", "3.1.1", "3.2", "3.3"] = "3.3"
    where: list[MunsellContains | MunsellGroup] = []
    min_subset_size: int = 2

    def load(self) -> WeightedDataset:
        where = load_munsell.create_query(self.where)
        datasets = load_munsell(
            where=where,
            version=self.version,
            min_subset_size=self.min_subset_size,
        )
        return WeightedDataset(
            datasets=datasets, weight=self.weight, name=self.name
        )

    @property
    def display_name(self):
        custom = self.where or self.min_subset_size != 2
        return f"Munsell-{self.version}{'*' if custom else ''}"


class DatasetRit_dupont(BaseDataset):
    name: Literal["rit_dupont"] = "rit_dupont"
    display_name: ClassVar[str] = "RIT-DuPont"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_rit_dupont(), weight=self.weight, name=self.name
        )


class DatasetWitt(BaseDataset):
    name: Literal["witt"] = "witt"
    display_name: ClassVar[str] = "Witt"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_witt(), weight=self.weight, name=self.name
        )


class DatasetCombvd(BaseDataset):
    name: Literal["combvd"]
    display_name: ClassVar[str] = "COMBVD"

    # BFD-P, Leeds, RIT-DuPont, Witt.
    def load(self) -> WeightedDataset:
        datasets = (
            load_witt() + load_leeds() + load_bfd_p() + load_rit_dupont()
        )
        return WeightedDataset(
            datasets=datasets, weight=self.weight, name=self.name
        )


DatasetConfig = Annotated[
    DatasetBfd_p
    | DatasetCombvd
    | DatasetFairchild_chen
    | DatasetHung_berns
    | DatasetIlluminants
    | DatasetLeeds
    | DatasetLuo_rigg
    | DatasetMacadam_1942
    | DatasetMacadam_1974
    | DatasetMunsell
    | DatasetRit_dupont
    | DatasetWitt,
    Field(..., discriminator="name"),
]
