from json import loads
from typing import Literal, TypedDict
from pathlib import Path


class Sensitivity(TypedDict):
    name: Literal[
        "CIE 1931 2-degree Standard Observer",
        "CIE 1964 10-degree Standard Observer",
    ]
    lambda_nm: tuple[int, int, int]
    xyz: list[list[float]]


def load(observer: Literal["cie-1931-2", "cie-1964-10"]) -> Sensitivity:
    assert observer in ("cie-1931-2", "cie-1964-10")
    data = loads(Path(__file__).with_name(f"{observer}.json").read_text())
    return data
