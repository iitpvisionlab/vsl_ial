from __future__ import annotations
import unittest
from vsl_ial.stress import (
    stress,
    group_stress,
    weighted_group_stress,
    mean_stress,
    calc_k,
    cv,
    pf3,
)
import numpy as np


a = np.asarray(
    (1.3, 5.0, 10.0, 3.0, 0.12, 10.0, 32.4, 1, 8.0, 0.4, 0.454),
    dtype=np.float64,
)
b = np.asarray(
    (2.3, 3.0, 13.0, 2.0, 0.52, 33.0, 2.0, 17.0, 0.4, 0.30, 0.7),
    dtype=np.float64,
)


class TestCaseS(unittest.TestCase):
    def test_calc_k(self):
        np.testing.assert_almost_equal(calc_k(a, b, 2), 0.42158523460540476)
        np.testing.assert_almost_equal(calc_k(a, b, 1), 0.0617283950617284)
        np.testing.assert_almost_equal(calc_k(a, a + 1.0, 1), 1.1)

    def test_stress(self):
        np.testing.assert_almost_equal(stress(a, a * 0.75), 0.0)
        np.testing.assert_almost_equal(stress(a * -2.0, a), 0.0)
        np.testing.assert_almost_equal(stress(a, a * 0.75, 1), 0.0)
        np.testing.assert_almost_equal(stress(a * -2.0, a, 1), 0.0)

    def test_group_stress(self):
        np.testing.assert_almost_equal(
            group_stress([a, a * 2.0], [a * 0.75, a], 2), 0.0
        )
        np.testing.assert_almost_equal(
            group_stress([a, a * 2.0], [a * 0.75, a], 1), 0.0
        )
        np.testing.assert_almost_equal(
            group_stress([a, b, a + 1.0], [b, a, b + 0.3]), 0.9143935566472584
        )
        np.testing.assert_almost_equal(
            weighted_group_stress(
                [a, b, a + 1.0],
                [b, a, b + 0.3],
                weights=[0.1, 0.3, 1.1],
                ord=2,
            ),
            0.9075584440794656,
        )

    def test_cv(self):
        np.testing.assert_almost_equal(
            cv([a, b, a + 1.0], [b, a, b + 0.3]),
            153.00545188807772,
        )

    def test_pf3(self):
        np.testing.assert_almost_equal(
            pf3([a, b, a + 1.0], [b, a, b + 0.3]),
            135.6415311204017,
        )

    def test_mean_stress(self):
        np.testing.assert_almost_equal(
            mean_stress([a, b, a + 1.0], [b, a, b + 0.3]),
            0.9147436796972936,
        )
