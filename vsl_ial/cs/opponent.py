from __future__ import annotations
from . import CS, FArray
import numpy as np


class Opponent(CS):
    """
    https://sid.onlinelibrary.wiley.com/doi/pdf/10.1889/1.1985127?casa_token=B5Gc5J-fFsgAAAAA:rgihZLGZbu2AnpZq6_Gxj78PG4fA0Eh4sPpW5A8Pkg4WJ-7J_rkyJhcxGUjMgS02XhFV8JGppZ22CkrZ
    """

    M = np.array(
        (
            (0.279, -0.449, 0.086),
            (0.72, 0.29, -0.59),
            (-0.107, -0.077, 0.501),
        ),
        dtype=np.float64,
    )

    Minv = np.array(
        (
            (0.62655450425, 1.369855450124, 1.505650754905),
            (-1.867177597834, 0.93475582413, 1.421323771758),
            (-0.15315637341, 0.436229005232, 2.53602108024),
        ),
        dtype=np.float64,
    )

    def __init__(self, illuminant_xyz: FArray | None = None):
        assert illuminant_xyz is None
        super().__init__(illuminant_xyz)

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        return np.tensordot(color, self.M, axes=1)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        return np.tensordot(color, self.Minv, axes=1)
