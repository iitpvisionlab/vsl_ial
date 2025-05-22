from . import CS, FArray
import numpy as np


class ICaCb(CS):
    rgb_matrix = np.array(
        (
            (0.37613, -0.21649, 0.02567),
            (0.70431, 1.14744, 0.16713),
            (-0.05675, 0.05356, 0.74235),
        ),
        dtype=np.float64,
    )

    rgb_matrix_inv = np.array(
        (
            (1.9380097630323718, 0.3726932255022942, -0.15092197668921598),
            (-1.2240118236920852, 0.6453747900837897, -0.10297178575136787),
            (0.2364654507025462, -0.018072247874496617, 1.3429653757226783),
        ),
        dtype=np.float64,
    )

    lCaCb_matrix = np.array(
        (
            (0.4949, 4.2854, 0.3605),
            (0.5037, -4.5462, 1.1499),
            (-0.0015, 0.2609, -1.5105),
        ),
        dtype=np.float64,
    )

    lCaCb_matrix_inv = np.array(
        (
            (1.0028987160875324, 1.0029177996452858, 1.0028468487002118),
            (0.11592986443747855, -0.11408071717162871, -0.059178219494170695),
            (0.019027973225823804, -0.02070045402816674, -0.6732498296849252),
        ),
        dtype=np.float64,
    )

    def from_XYZ(self, src: CS, color: FArray) -> FArray:
        n = 2610.0 / (2**14)
        m = 2523.0 / (2**5)
        c_1 = 3424.0 / (2**12)
        c_2 = 2413.0 / (2**7)
        c_3 = 2392.0 / (2**7)

        rgb_points = np.tensordot(color, self.rgb_matrix, axes=1)

        rgb_tmp_points = (
            (c_1 + c_2 * ((rgb_points / 10000) ** n))
            / (1.0 + c_3 * ((rgb_points / 10000) ** n))
        ) ** m
        return np.tensordot(rgb_tmp_points, self.lCaCb_matrix, axes=1)

    def to_XYZ(self, src: CS, color: FArray) -> FArray:
        n = 2610.0 / (2**14)
        m = 2523.0 / (2**5)
        c_1 = 3424.0 / (2**12)
        c_2 = 2413.0 / (2**7)
        c_3 = 2392.0 / (2**7)

        step1 = np.tensordot(color, self.lCaCb_matrix_inv, axes=1)
        rgb_tmp_points = (
            10000.0**n
            * (-c_1 + step1 ** (1 / m))
            / (c_2 - c_3 * step1 ** (1 / m))
        ) ** (1 / n)
        return np.tensordot(rgb_tmp_points, self.rgb_matrix_inv, axes=1)
