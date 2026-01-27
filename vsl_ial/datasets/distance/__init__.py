from __future__ import annotations
from typing import (
    Literal,
    NamedTuple,
    Callable,
    TypeAlias,
    Sequence,
    Iterator,
    TypeVar,
)
from pathlib import Path
import json
import numpy as np


dataset_root = Path(__file__).parent

T = TypeVar("T")


class DistanceDataset(NamedTuple):
    name: str
    L_A: float  # the relative tristimulus values of the sample in the source condition
    Y_b: float
    c: float  # the impact of surround
    Nc: float  # achromatic induction factor
    F: float  # actor for degree of adaptation
    illuminant: np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]
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
            Nc=data.get("Nc"),
            F=data.get("F"),
            illuminant=np.array(data["reference_white"], dtype=np.float64)
            * 0.01,
            xyz=np.array(data["xyz"], dtype=np.float64) * 0.01,
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


class MunsellContains(NamedTuple):
    hue: Sequence[MunsellHue] = ()
    chroma: Sequence[int] | MunsellLimit = ()
    value: Sequence[int] | MunsellLimit = ()

    def create(self) -> WhereFunction:
        return self.check

    def check(self, group: Literal["HC", "VC", "HV"], row: MunsellRow) -> bool:
        return (
            (not self.hue or row.H_hue in self.hue)
            and (not self.value or row.V in self.value)
            and (not self.chroma or row.C in self.chroma)
        )


class MunsellGroup(NamedTuple):
    group: Literal["HV", "VC", "HC"]
    match: Literal["any"] | MunsellContains = "any"

    def create(self) -> WhereFunction:
        if self.match == "any":

            def f(group: Literal["HC", "VC", "HV"], row: MunsellRow) -> bool:
                return group == self.group

        else:
            check: WhereFunction = self.match.check

            def f(group: Literal["HC", "VC", "HV"], row: MunsellRow) -> bool:
                return group == self.group and check(group, row)

        return f


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
    def key_h_next(h: tuple[int, int]):
        h_index, h_value = h
        if h_value != 10.0:
            return h_index, h_value + 2.5
        if h_index != 9:
            return h_index + 1.0, 2.5
        return 0, 2.5  # wrap

    @staticmethod
    def create_query(
        conditions: list[MunsellContains | MunsellGroup],
    ) -> WhereFunction:
        if not conditions:
            return lambda _1, _2: True
        fns = [condition.create() for condition in conditions]
        return lambda group, row: any(fn(group, row) for fn in fns)

    @staticmethod
    def key_v(row: MunsellRow):
        return row.V

    @staticmethod
    def key_v_next(V: int):
        return V + 1

    @staticmethod
    def key_c(row: MunsellRow):
        return row.C

    @staticmethod
    def key_c_next(C: int):
        return C + 2

    @staticmethod
    def split_rows(
        rows: list[MunsellRow],
        get_key: Callable[[MunsellRow], T],
        next_key: Callable[[T], T],
    ) -> Iterator[list[MunsellRow]]:
        it = iter(rows)
        prev_row = next(it)
        first_key = prev_key = get_key(prev_row)
        first_group = cur_group = [prev_row]
        for row in it:
            key = get_key(row)
            if next_key(prev_key) == key:
                cur_group.append(row)
            else:
                yield cur_group
                cur_group = [row]
            prev_row, prev_key = row, key
        if next_key(prev_key) == first_key:
            if first_group is cur_group:
                first_group.append(rows[0])
                yield cur_group
            else:
                first_group[:0] = cur_group
        else:
            yield cur_group

    def __new__(
        cls,
        where: WhereFunction | None = None,
        version: Literal["2.0", "3.0", "3.1.0", "3.1.1", "3.2", "3.3"] = "3.3",
        min_subset_size: int = 2,
    ):
        from collections import defaultdict

        version_d = version.replace(".", "-")
        csv = (
            dataset_root / "order" / "mrr-revised" / f"munsell_{version_d}.csv"
        )
        groups_hv = defaultdict[str, list[MunsellRow]](list)
        groups_vc = defaultdict[str, list[MunsellRow]](list)
        groups_hc = defaultdict[str, list[MunsellRow]](list)

        with csv.open() as f:
            next(f)
            for line in f:
                row = MunsellRow.from_line(line)
                if where is None or where("HV", row):
                    groups_hv[f"h={row.H}_v={row.V}"].append(row)
                if row.C != 0 and (where is None or where("VC", row)):
                    groups_vc[f"v={row.V}_c={row.C}"].append(row)
                if where is None or where("HC", row):
                    groups_hc[f"h={row.H}_c={row.C}"].append(row)

        ret: list[DistanceDataset] = []
        for groups, key, key_next in (
            (groups_hv, cls.key_c, cls.key_c_next),
            (groups_vc, cls.key_h, cls.key_h_next),
            (groups_hc, cls.key_v, cls.key_v_next),
        ):
            for name, group in groups.items():
                group.sort(key=key)
                subsets: list[list[MunsellRow]] = []
                for group in list(cls.split_rows(group, key, key_next)):
                    subsets.append(group)
                if subsets:
                    if (
                        sum(len(subset) for subset in subsets)
                        > min_subset_size
                    ):  # filter by number of pairs. > and not >= !
                        ret.append(
                            cls.subsets_as_dataset(
                                f"{version}-{name}", subsets
                            )
                        )
        return ret

    @staticmethod
    def subsets_as_dataset(
        key: str, subsets: list[list[MunsellRow]]
    ) -> DistanceDataset:
        from vsl_ial.cs import whitepoints_cie1931
        from vsl_ial.cs.ciexyy import CIExyY

        pairs: list[tuple[int, int]] = []
        shift = 0
        for rows in subsets:
            n = len(rows)
            pairs.extend(
                zip(
                    range(shift + 0, shift + n - 1),
                    range(shift + 1, shift + n),
                )
            )
            shift += len(rows)

        xyY = np.asarray(
            [(row.x, row.y, row.Y * 0.01) for rows in subsets for row in rows],
            dtype=np.float64,
        )
        xyz = CIExyY(None).to_XYZ(None, xyY)

        return DistanceDataset(
            name=f"munsell-{key}",
            L_A=64.0,
            Y_b=20.0,
            c=0.69,
            Nc=1.0,
            F=1.0,
            illuminant=whitepoints_cie1931.C,
            # 1.0 is a perceptive step, we don't know its exact value
            dv=np.full(shape=(len(pairs),), fill_value=np.float64(1.0)),
            pairs=pairs,
            xyz=xyz,
        )
