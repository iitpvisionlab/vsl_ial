from __future__ import annotations
from pathlib import Path

from typing import Literal, Annotated, ClassVar, Any
from pydantic import Field
import json5
from ._base import StrictModel
from typing import NamedTuple
from vsl_ial import FArray
import numpy as np


class Dataset(NamedTuple):
    name: str
    L_A: float  # the relative tristimulus values of the sample in the source condition
    Y_b: float
    c: float  # the impact of surround
    Nc: float | None  # achromatic induction factor
    F: float | None  # actor for degree of adaptation
    illuminant: tuple[float, float, float]
    xyz: FArray  # list[tuple[float, float, float]]
    pairs: list[tuple[int, int]]
    dv: list[float]


dataset_root = Path(__file__).parents[1] / "datasets" / "data"


class BaseDataset(StrictModel):
    # name: ClassVar[Literal["bfd_p", "ebner_fairchild", "fairchild_chen", "hung_berns", "illuminants", "leeds", "luo_rigg", "macadam_1942", "macadam_1974", "mrr", "munsell", "observers", "rit_dupont", "witt", "xiao", "combvd"]]

    def load(self) -> list[Dataset]:
        raise NotImplementedError(self)
        # assert isinstance(self.name, str)
        # paths: list[Path] = list((dataset_root / self.name).glob("*.json"))
        # if len(paths) == 0:
        #     return json5.loads(path.read_text(paths[0]))
        # ret = {}
        # for path in paths:
        #     ret[path.name] = json5.loads(path.read_text())
        # return ret


class JsonDataset(BaseDataset):
    def load(self):
        assert isinstance(self.name, str)
        return [self.load_json(self.path, self.name)]

    @staticmethod
    def load_json(path: Path, name: str) -> Dataset:
        text = path.read_text()
        data = json5.loads(text)
        assert len(data["dv"]) == len(data["pairs"]), "dataset error"
        return Dataset(
            name=name,
            L_A=data["L_A"],
            Y_b=data["Y_b"],
            c=data["c"],
            Nc=None,
            F=None,
            illuminant=data["reference_white"],
            xyz=np.asarray(data["xyz"], dtype=np.float64),
            pairs=data["pairs"],
            dv=np.asarray(data["dv"], dtype=np.float64),
        )


class DatasetBfd_p(JsonDataset):
    name: Literal["bfd_p"] = "bfd_p"
    subsets: list[Literal["c", "d65", "m"]] = ["c", "d65", "m"]

    def load(self):
        assert isinstance(self.name, str)
        datasets: list[Dataset] = []
        for subset in self.subsets:
            path = dataset_root / "bfd_p" / f"bfd-{subset}.json"
            datasets.append(self.load_json(path, f"{self.name}-{subset}"))
        return datasets


class DatasetEbner_fairchild(BaseDataset):
    name: Literal["ebner_fairchild"]


class DatasetFairchild_chen(BaseDataset):
    name: Literal["fairchild_chen"]


class DatasetHung_berns(BaseDataset):
    name: Literal["hung_berns"]


class DatasetIlluminants(BaseDataset):
    name: Literal["illuminants"]


class DatasetLeeds(JsonDataset):
    name: Literal["leeds"] = "leeds"

    @property
    def path(self):
        return dataset_root / "leeds" / "leeds.json"


class DatasetLuo_rigg(BaseDataset):
    name: Literal["luo_rigg"]


class DatasetMacadam_1942(BaseDataset):
    name: Literal["macadam_1942"]


class DatasetMacadam_1974(BaseDataset):
    name: Literal["macadam_1974"]


class DatasetMrrRevised(BaseDataset):
    name: Literal["mrr"]


class DatasetMunsell(BaseDataset):
    name: Literal["munsell"]


class DatasetObservers(BaseDataset):
    name: Literal["observers"]


class DatasetRit_dupont(JsonDataset):
    name: Literal["rit_dupont"] = "rit_dupont"

    @property
    def path(self):
        return dataset_root / "rit_dupont" / "rit-dupont.json"


class DatasetWitt(JsonDataset):
    name: Literal["witt"] = "witt"

    @property
    def path(self):
        return dataset_root / "witt" / "witt.json"


class DatasetXiao(BaseDataset):
    name: Literal["xiao"]


class DatasetCombvd(BaseDataset):
    name: Literal["combvd"]

    # BFD-P, Leeds, RIT-DuPont, Witt.
    def load(self) -> list[Dataset]:
        return [
            *DatasetWitt().load(),
            *DatasetLeeds().load(),
            *DatasetBfd_p().load(),
            *DatasetRit_dupont().load(),
        ]


DatasetConfig = Annotated[
    DatasetBfd_p
    | DatasetCombvd
    | DatasetEbner_fairchild
    | DatasetFairchild_chen
    | DatasetHung_berns
    | DatasetIlluminants
    | DatasetLeeds
    | DatasetLuo_rigg
    | DatasetMacadam_1942
    | DatasetMacadam_1974
    | DatasetMrrRevised
    | DatasetMunsell
    | DatasetObservers
    | DatasetRit_dupont
    | DatasetWitt
    | DatasetXiao,
    Field(..., discriminator="name"),
]
