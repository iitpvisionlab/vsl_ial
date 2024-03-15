from __future__ import annotations
from . import CS, FArray


class XYZ(CS):
    def __init__(self, illuminant_xyz: FArray | None = None):
        assert illuminant_xyz is None
        super().__init__(illuminant_xyz)
