from __future__ import annotations
from typing import Literal, Annotated

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
    subsets: list[Literal["c", "d65", "m"]] = ["c", "d65", "m"]

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_bfd_p(subsets=self.subsets),
            weight=self.weight,
            name=self.name,
        )


class DatasetFairchild_chen(BaseDataset):
    name: Literal["fairchild_chen"]


class DatasetHung_berns(BaseDataset):
    name: Literal["hung_berns"]


class DatasetIlluminants(BaseDataset):
    name: Literal["illuminants"]


class DatasetLeeds(BaseDataset):
    name: Literal["leeds"] = "leeds"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_leeds(), weight=self.weight, name=self.name
        )


class DatasetLuo_rigg(BaseDataset):
    name: Literal["luo_rigg"]


class DatasetMacadam_1942(BaseDataset):
    name: Literal["macadam_1942"]


class DatasetMacadam_1974(BaseDataset):
    name: Literal["macadam_1974"]


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


class DatasetRit_dupont(BaseDataset):
    name: Literal["rit_dupont"] = "rit_dupont"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_rit_dupont(), weight=self.weight, name=self.name
        )


class DatasetWitt(BaseDataset):
    name: Literal["witt"] = "witt"

    def load(self) -> WeightedDataset:
        return WeightedDataset(
            datasets=load_witt(), weight=self.weight, name=self.name
        )


class DatasetCombvd(BaseDataset):
    name: Literal["combvd"]

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
