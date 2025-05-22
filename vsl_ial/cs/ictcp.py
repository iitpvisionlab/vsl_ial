from __future__ import annotations
from . import CS, FArray
import numpy as np


class ICtCp(CS):
    lms_matrix = np.array(
        (
            (0.3593, -0.1921, 0.0071),
            (0.6976, 1.1005, 0.0748),
            (-0.0359, 0.0754, 0.8433),
        ),
        dtype=np.float64,
    )

    lms_matrix_inv = np.array(
        (
            (2.0700345210961406, 0.3647497844879312, -0.04978125101325728),
            (-1.326231182012956, 0.6805456704366378, -0.04919788302664357),
            (0.20670232471377717, -0.04532032051204287, 1.188097205583817),
        ),
        dtype=np.float64,
    )

    ictcp_matrix = (
        np.array(
            [[2048, 6610, 17933], [2048, -13613, -17933], [0, 7003, -543]],
            dtype=np.float64,
        )
        / 4096
    ) * [720.0, 360.0, 720.0]

    ictcp_matrix_inv = np.array(
        [
            (1.0145055084292036, 0.9854944915707965, 0.9581115383458825),
            (0.008484158578611704, -0.008484158578611704, 0.5603919550285215),
            (0.10941908384165325, -0.10941908384165325, -0.31597631479790855),
        ],
        dtype=np.float64,
    ) / [[720.0], [360.0], [720.0]]

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        m_1 = 0.1593017578125
        m_2 = 78.84375
        c_2 = 18.8515625
        c_3 = 18.6875
        c_1 = c_3 - c_2 + 1
        lms = np.tensordot(color, self.lms_matrix, axes=1)
        lms_2 = ((c_1 + (c_2 * lms**m_1)) / (1.0 + (c_3 * lms**m_1))) ** m_2

        return np.tensordot(lms_2, self.ictcp_matrix, axes=1)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        m_1 = 1.0 / 0.1593017578125
        m_2 = 1.0 / 78.84375
        c_2 = 18.8515625
        c_3 = 18.6875
        c_1 = c_3 - c_2 + 1
        step1 = np.tensordot(color, self.ictcp_matrix_inv, axes=1)
        step2 = ((-c_1 + step1**m_2) / (c_2 - c_3 * step1**m_2)) ** (m_1)
        return np.tensordot(step2, self.lms_matrix_inv, axes=1)
