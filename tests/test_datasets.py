from __future__ import annotations
from unittest import TestCase
from vsl_ial.datasets.sensitivities import load as load_sensitivities
from vsl_ial.datasets.distance import (
    load_munsell,
    MunsellGroup,
    MunsellContains,
    MunsellLimit,
    load_bfd_p,
    load_rit_dupont,
    load_witt,
    load_leeds,
    load_combvd,
)


class TestLoadIntensities(TestCase):
    def test_load_cie_1931_2(self):
        dataset = load_sensitivities("cie-1931-2")
        self.assertIn("xyz", dataset)

    def test_load_cie_1964_10(self):
        dataset = load_sensitivities("cie-1964-10")
        self.assertIn("xyz", dataset)


class TestLoadDistances(TestCase):
    def test_load_load_bfd_p(self):
        self.assertEqual(len(load_bfd_p()), 3)

    def test_load_rit_dupont(self):
        self.assertEqual(len(load_rit_dupont()), 1)

    def test_load_witt(self):
        self.assertEqual(len(load_witt()), 1)

    def test_load_leeds(self):
        self.assertEqual(len(load_leeds()), 1)

    def test_load_combvd(self):
        self.assertEqual(len(load_combvd()), 6)


class TestLoadMunsell(TestCase):
    def test_load_2_0(self):
        self.assertEqual(len(load_munsell(version="2.0")), 910)

    def test_load_3_2(self):
        self.assertEqual(len(load_munsell(version="3.2")), 952)

    def test_load_3_3(self):
        self.assertEqual(len(load_munsell(version="3.3")), 952)

    def test_where_0(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="HC",
                        match=MunsellContains(value=MunsellLimit(max=0)),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 0)

    def test_where_any(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [MunsellGroup(group="HC", match="any")]
            ),
        )
        self.assertEqual(len(dataset), 422)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [MunsellGroup(group="VC", match="any")]
            ),
        )
        self.assertEqual(len(dataset), 133)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [MunsellGroup(group="HV", match="any")]
            ),
        )
        self.assertEqual(len(dataset), 397)

        assert 422 + 133 + 397 == 952

    def test_where_hc_limit(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="HC",
                        match=MunsellContains(value=MunsellLimit(max=5)),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 358)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="HC",
                        match=MunsellContains(value=MunsellLimit(min=6)),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 358)

    def test_where_hv_limit(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="HV",
                        match=MunsellContains(chroma=MunsellLimit(max=5)),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 396)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="HV",
                        match=MunsellContains(chroma=MunsellLimit(min=6)),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 328)

    def test_where_vc_limit(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="VC",
                        match=MunsellContains(hue=["B", "BG", "G", "GY", "P"]),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 127)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="VC",
                        match=MunsellContains(
                            hue=["PB", "R", "RP", "Y", "YR"]
                        ),
                    )
                ]
            ),
        )
        self.assertEqual(len(dataset), 109)

    def test_where_vc_contains(self):
        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [
                    MunsellGroup(
                        group="VC",
                        match=MunsellContains(hue=["B", "BG", "G", "GY", "P"]),
                    ),
                    MunsellContains(chroma=[8, 10, 18]),
                ]
            ),
        )
        self.assertEqual(len(dataset), 543)
