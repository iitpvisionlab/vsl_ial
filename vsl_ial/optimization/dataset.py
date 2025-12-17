from __future__ import annotations
from pathlib import Path
from typing import Literal, Annotated, ClassVar, Any, Callable, TypeAlias

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
    weight: float


dataset_root = Path(__file__).parents[1] / "datasets"


class BaseDataset(StrictModel):
    weight: float  # = 1.0
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

    def load_json(self, path: Path, name: str) -> Dataset:
        text = path.read_text()
        data = json5.loads(text)
        assert len(data["dv"]) == len(data["pairs"]), "dataset error"
        return Dataset(
            name=name,
            L_A=data["L_A"],
            Y_b=data["Y_b"],
            c=data["c"],
            Nc=None,
            F=data.get("F"),
            illuminant=data["reference_white"],
            xyz=np.asarray(data["xyz"], dtype=np.float64),
            pairs=data["pairs"],
            dv=np.asarray(data["dv"], dtype=np.float64),
            weight=self.weight,
        )


class DatasetBfd_p(JsonDataset):
    name: Literal["bfd_p"] = "bfd_p"
    subsets: list[Literal["c", "d65", "m"]] = ["c", "d65", "m"]

    def load(self):
        assert isinstance(self.name, str)
        datasets: list[Dataset] = []
        for subset in self.subsets:
            path = (
                dataset_root
                / "distance"
                / "combvd"
                / self.name
                / f"bfd-{subset}.json"
            )
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
        return dataset_root / "distance" / "combvd" / "leeds.json"


class DatasetLuo_rigg(BaseDataset):
    name: Literal["luo_rigg"]


class DatasetMacadam_1942(BaseDataset):
    name: Literal["macadam_1942"]


class DatasetMacadam_1974(BaseDataset):
    name: Literal["macadam_1974"]


# class DatasetMrrRevised(BaseDataset):
#     name: Literal["mrr"]


class DatasetMunsell(BaseDataset):
    name: Literal["munsell"]
    version: Literal["2.0", "3.2"] = "3.2"

    class Row(NamedTuple):
        H: str
        H_color: Literal["R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"]
        H_value: float
        V: int
        C: int
        x: float
        y: float
        Y: float

        @classmethod
        def from_line(cls, line: str):
            H, V, C, x, y, Y = line.split(",")
            idx = H.index(".") + 2
            H_value = float(H[:idx])
            H_color = H[idx:]
            return cls(
                H=H,
                H_color=H_color,
                H_value=H_value,
                V=int(V),
                C=int(C),
                x=float(x),
                y=float(y),
                Y=float(Y),
            )

    # class Row2(NamedTuple):
    #     row: Row
    #     order: Literal["hv", "vc", "hc"]

    #     def key_hv(self):
    #         return self.row.h

    @staticmethod
    def key_h(
        row: Row,
        order: tuple[str, ...] = (
            "R",
            "YR",
            "Y",
            "GY",
            "G",
            "BG",
            "B",
            "PB",
            "P",
            "RP",
        ),
    ):
        return order.index(row.H_color), row.H_value

    @staticmethod
    def key_v(row: Row):
        return row.V

    @staticmethod
    def key_c(row: Row):
        return row.C

    def load(self) -> list[Dataset]:
        from collections import defaultdict

        Key: TypeAlias = tuple[str, Callable[[DatasetMunsell.Row], Any]]

        version = self.version.replace(".", "-")
        csv = (
            dataset_root
            / "distance"
            / "order"
            / "mrr-revised"
            / f"munsell_{version}.csv"
        )
        groups = defaultdict[Key, self.Row](list)

        with csv.open() as f:
            next(f)
            for line in f:
                row = self.Row.from_line(line)
                # groups[(f"h={row.H}_v={row.V}", self.key_c)].append(row)
                # if row.C != 0:
                #     groups[(f"v={row.V}_c={row.C}", self.key_h)].append(row)
                if 18 <= row.C <= 32:
                    groups[(f"h={row.H}_c={row.C}", self.key_v)].append(row)

        ret: list[Dataset] = []
        for key, group in groups.items():
            if len(group) > 1:
                group.sort(key=key[1])
                ret.append(
                    self.group_as_dataset(f"{self.version}-{key[0]}", group)
                )
        return ret

    def group_as_dataset(self, key: str, rows: list[Row]) -> Dataset:
        from vsl_ial.cs import whitepoints_cie1931
        from vsl_ial.cs.ciexyy import CIExyY

        n = len(rows) - 1

        xyy = np.asarray(
            [(row.x, row.y, row.Y) for row in rows], dtype=np.float64
        )
        xyz = CIExyY(None).to_XYZ(None, xyy)

        return Dataset(
            name=f"munsell-{key}",
            L_A=64.0,
            Y_b=20.0,
            c=0.69,
            Nc=1.0,
            F=1.0,
            illuminant=whitepoints_cie1931.C,
            dv=np.asarray([1.0] * n, dtype=np.float64),
            pairs=list(zip(range(n), range(1, n + 1))),
            xyz=xyz,
            weight=self.weight,
        )


class DatasetObservers(BaseDataset):
    name: Literal["observers"]


class DatasetRit_dupont(JsonDataset):
    name: Literal["rit_dupont"] = "rit_dupont"

    @property
    def path(self):
        return dataset_root / "distance" / "combvd" / "rit-dupont.json"


class DatasetWitt(JsonDataset):
    name: Literal["witt"] = "witt"

    @property
    def path(self):
        return dataset_root / "distance" / "combvd" / "witt.json"


class DatasetXiao(BaseDataset):
    name: Literal["xiao"]


class DatasetCombvd(BaseDataset):
    name: Literal["combvd"]

    # BFD-P, Leeds, RIT-DuPont, Witt.
    def load(self) -> list[Dataset]:
        return [
            *DatasetWitt(weight=self.weight).load(),
            *DatasetLeeds(weight=self.weight).load(),
            *DatasetBfd_p(weight=self.weight).load(),
            *DatasetRit_dupont(weight=self.weight).load(),
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
    | DatasetMunsell
    | DatasetObservers
    | DatasetRit_dupont
    | DatasetWitt
    | DatasetXiao,
    Field(..., discriminator="name"),
]
