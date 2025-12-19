from __future__ import annotations
from typing import (
    Literal,
    NamedTuple,
    Callable,
    TypeAlias,
    Sequence,
)
from pathlib import Path
import json
import numpy as np


dataset_root = Path(__file__).parent


class DistanceDataset(NamedTuple):
    name: str
    L_A: float  # the relative tristimulus values of the sample in the source condition
    Y_b: float
    c: float  # the impact of surround
    Nc: float | None  # achromatic induction factor
    F: float | None  # actor for degree of adaptation
    illuminant: tuple[float, float, float]
    xyz: np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
    pairs: list[tuple[int, int]]
    dv: np.ndarray[tuple[int], np.dtype[np.float64]]

    @classmethod
    def load_json(cls, path: Path, name: str):
        text = path.read_text()
        data = json.loads(text)
        assert len(data["dv"]) == len(data["pairs"]), f"dataset error {path}"
        return cls(
            name=name,
            L_A=data["L_A"],
            Y_b=data["Y_b"],
            c=data["c"],
            Nc=None,
            F=data.get("F"),
            illuminant=data["reference_white"],
            xyz=np.array(data["xyz"], dtype=np.float64),
            pairs=data["pairs"],
            dv=np.array(data["dv"], dtype=np.float64),
        )


def load_bfd_p(subsets: list[Literal["c", "d65", "m"]] = ["c", "d65", "m"]):
    return [
        DistanceDataset.load_json(
            dataset_root / "combvd" / "bfd_p" / f"bfd-{subset}.json",
            f"bfd_p-{subset}",
        )
        for subset in subsets
    ]


def load_leeds():
    return [
        DistanceDataset.load_json(
            dataset_root / "combvd" / "leeds.json", "leeds"
        )
    ]


def load_rit_dupont():
    return [
        DistanceDataset.load_json(
            dataset_root / "combvd" / "rit-dupont.json", "rit-dupont"
        )
    ]


def load_witt():
    return [
        DistanceDataset.load_json(
            dataset_root / "combvd" / "witt.json", "witt"
        )
    ]


def load_combvd():
    return [*load_witt(), *load_leeds(), *load_bfd_p(), *load_rit_dupont()]


MunsellHue: TypeAlias = Literal[
    "R", "YR", "Y", "GY", "G", "BG", "B", "PB", "P", "RP"
]


class MunsellLimit(NamedTuple):
    min: int = -1000
    max: int = +1000

    def __call__(self, value: int | MunsellHue) -> bool:
        assert isinstance(value, int), f"Invalid type for comparison {value}"
        return self.min <= value <= self.max

    __contains__ = __call__  # type: ignore


class MunsellRow(NamedTuple):
    H: str
    H_hue: MunsellHue
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
        H_hue = H[idx:]
        return cls(
            H=H,
            H_hue=H_hue,
            H_value=H_value,
            V=int(V),
            C=int(C),
            x=float(x),
            y=float(y),
            Y=float(Y),
        )


WhereFunction: TypeAlias = Callable[
    [Literal["HV", "VC", "HC"], MunsellRow], bool
]


class MunsellConst(NamedTuple):
    const: Literal["HV", "VC", "HC"]
    match: Literal["any"] | MunsellLimit | list[MunsellHue]

    def create(self) -> WhereFunction:
        check: Callable[[int | MunsellHue], bool]
        match self.match:
            case MunsellLimit() as lim:
                check = lim.__contains__
            case "any":
                check = lambda _: True
            case list() as hues:
                check = lambda hue: hue in hues
            case _:
                raise ValueError(f"Unsupported match {self.match}")

        def f(const: Literal["HC", "VC", "HV"], row: MunsellRow) -> bool:
            if const != self.const:
                return False
            if const == "HC":
                return check(row.V)
            if const == "HV":
                return check(row.C)
            return check(row.H_hue)

        return f


class MunsellContains(NamedTuple):
    hue: Sequence[MunsellHue] = ()
    chroma: Sequence[int] | MunsellLimit = ()
    value: Sequence[int] | MunsellLimit = ()

    def create(self) -> WhereFunction:
        return self.check

    def check(self, const: Literal["HC", "VC", "HV"], row: MunsellRow) -> bool:
        return (
            (not self.hue or row.H_hue in self.hue)
            and (not self.value or row.V in self.value)
            and (not self.chroma or row.C in self.chroma)
        )


class load_munsell:
    @staticmethod
    def key_h(
        row: MunsellRow,
        order: Callable[[MunsellHue], int] = (
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
        ).index,
    ):
        return order(row.H_hue), row.H_value

    @staticmethod
    def create_query(
        conditions: list[MunsellContains | MunsellConst],
    ) -> WhereFunction:
        if not conditions:
            return lambda _1, _2: True
        fns = [condition.create() for condition in conditions]
        return lambda const, row: all(fn(const, row) for fn in fns)

    @staticmethod
    def key_v(row: MunsellRow):
        return row.V

    @staticmethod
    def key_c(row: MunsellRow):
        return row.C

    def __new__(
        cls,
        where: WhereFunction | None = None,
        version: Literal["2.0", "3.2", "3.3"] = "3.3",
    ):
        from collections import defaultdict

        Key = tuple[str, Callable[[MunsellRow], int | tuple[int, float]]]

        version_d = version.replace(".", "-")
        csv = (
            dataset_root / "order" / "mrr-revised" / f"munsell_{version_d}.csv"
        )
        groups = defaultdict[Key, list[MunsellRow]](list)

        with csv.open() as f:
            next(f)
            for line in f:
                row = MunsellRow.from_line(line)
                if where is None or where("HV", row):
                    groups[(f"h={row.H}_v={row.V}", cls.key_c)].append(row)
                if row.C != 0 and (where is None or where("VC", row)):
                    groups[(f"v={row.V}_c={row.C}", cls.key_h)].append(row)
                if where is None or where("HC", row):
                    groups[(f"h={row.H}_c={row.C}", cls.key_v)].append(row)

        ret: list[DistanceDataset] = []
        for key, group in groups.items():
            if len(group) > 1:
                group.sort(key=key[1])
                ret.append(cls.group_as_dataset(f"{version}-{key[0]}", group))
        return ret

    @staticmethod
    def group_as_dataset(key: str, rows: list[MunsellRow]) -> DistanceDataset:
        from vsl_ial.cs import whitepoints_cie1931
        from vsl_ial.cs.ciexyy import CIExyY

        n = len(rows) - 1

        xyy = np.asarray(
            [(row.x, row.y, row.Y) for row in rows], dtype=np.float64
        )
        xyz = CIExyY(None).to_XYZ(None, xyy)

        return DistanceDataset(
            name=f"munsell-{key}",
            L_A=64.0,
            Y_b=20.0,
            c=0.69,
            Nc=1.0,
            F=1.0,
            illuminant=whitepoints_cie1931.C.tolist(),
            dv=np.full(shape=(n,), fill_value=np.float64(1.0)),
            pairs=list(zip(range(n), range(1, n + 1))),
            xyz=xyz,
        )
