from __future__ import annotations
from typing import NamedTuple, Literal
from pathlib import Path
from json import loads
import numpy as np
from ....cs import whitepoints_cie1931


XYZArray = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
Color = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]


class ColorSet(NamedTuple):
    colors: XYZArray


# must implement `eval.cs.SceneInfo`
class ConstantHueDataset(NamedTuple):
    name: str
    illuminant: Color
    items: list[ColorSet]
    L_A: float
    Y_b: float
    F: float
    c: float
    Nc: float


def load_ebner_fairchild() -> ConstantHueDataset:
    path = Path(__file__).with_name("ebner_fairchild.json")
    data = loads(path.read_text())
    items = [
        ColorSet(
            colors=np.array([item["reference xyz"]] + item["same"]) * 0.01,
        )
        for item in data["data"]
    ]
    illuminant = np.array(data["white point"]) * 0.01
    return ConstantHueDataset(
        name="Ebner and Fairchild",
        items=items,
        illuminant=illuminant,
        L_A=14.2,  # colorio: 25 (Yb * 71 / 100), Safdar: 14
        Y_b=35,  # colorio: 35, Safdar: 20
        c=0.525,  # "dark"
        Nc=0.8,
        F=0.8,
    )


def load_hung_berns() -> ConstantHueDataset:
    path = Path(__file__).with_name("hung_berns.json")
    data = loads(path.read_text())
    items = [
        ColorSet(
            colors=np.array(
                (
                    items["Ref."],
                    items["1/4"],
                    items["2/4"],
                    items["3/4"],
                )
            )
            * 0.01,
            # reference_xyz=np.array(items["Ref."]) * 0.01,
        )
        for items in data["table 3"].values()
    ]

    return ConstantHueDataset(
        name="Hung and Berns",
        items=items,
        illuminant=whitepoints_cie1931.C,
        L_A=10,
        Y_b=20,
        c=0.525,
        Nc=0.8,
        F=0.8,
    )


def load_xiao_averages() -> ConstantHueDataset:
    path = Path(__file__).with_name("xiao_averages.json")
    data = loads(path.read_text())
    data.pop("neutral-gray")
    illuminant = np.array([98.0, 100.0, 139.7]) * 0.01
    items = [
        ColorSet(
            # reference_hue_angle=float(item["reference hue angle"]),
            # reference_xyz= None, #np.array(item["reference xyz"]) * 0.01,
            colors=np.array(item)
            * 0.01,
        )
        for item in data.values()
    ]
    return ConstantHueDataset(
        name="Xiao et al.",
        items=items,
        illuminant=illuminant,
        L_A=23,
        Y_b=20,
        c=0.59,  # "dim" # colorio: 0.59, Safdar: 0.525
        Nc=0.9,  # colorio: 0.9, Safdar: 0.8
        F=0.9,  # colorio: 0.9, Safdar: 0.8
    )
