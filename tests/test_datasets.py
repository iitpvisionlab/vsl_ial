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
        self.assertEqual(len(load_munsell(version="2.0")), 854)

    def test_load_3_2(self):
        self.assertEqual(len(load_munsell(version="3.2")), 887)

    def test_load_3_3(self):
        self.assertEqual(len(load_munsell(version="3.3")), 887)

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
        self.assertEqual(len(dataset), 393)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [MunsellGroup(group="VC", match="any")]
            ),
        )
        self.assertEqual(len(dataset), 122)

        dataset = load_munsell(
            version="3.3",
            where=load_munsell.create_query(
                [MunsellGroup(group="HV", match="any")]
            ),
        )
        self.assertEqual(len(dataset), 372)

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
        self.assertEqual(len(dataset), 314)

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
        self.assertEqual(len(dataset), 290)

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
        self.assertEqual(len(dataset), 99)

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
        self.assertEqual(len(dataset), 304)

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
        self.assertEqual(len(dataset), 117)

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
        self.assertEqual(len(dataset), 104)

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
        self.assertEqual(len(dataset), 361)


class TestEval(TestCase):
    REF = (
        "                                          group_stress                                           \n"
        "┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┓\n"
        "┃           ┃ COMBVD ┃ BFD-P D65 ┃ BFD-P M ┃ BFD-P C ┃ RIT-DuPont ┃ Witt  ┃ Leeds ┃ Munsell-3.3 ┃\n"
        "┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━┩\n"
        "│ CAM16-SCD │ 0.295  │ 0.254     │ 0.346   │ 0.283   │ 0.237      │ 0.306 │ 0.219 │ 0.218       │\n"
        "│ CAM16-UCS │ 0.305  │ 0.271     │ 0.35    │ 0.297   │ 0.206      │ 0.31  │ 0.245 │ 0.205       │\n"
        "│ CAM16-LCD │ 0.339  │ 0.311     │ 0.372   │ 0.366   │ 0.214      │ 0.372 │ 0.292 │ 0.217       │\n"
        "│ CAM02-SCD │ 0.296  │ 0.266     │ 0.338   │ 0.303   │ 0.244      │ 0.303 │ 0.221 │ 0.266       │\n"
        "│ CAM02-UCS │ 0.306  │ 0.28      │ 0.343   │ 0.321   │ 0.213      │ 0.305 │ 0.246 │ 0.267       │\n"
        "│ CAM02-LCD │ 0.339  │ 0.318     │ 0.366   │ 0.404   │ 0.223      │ 0.366 │ 0.296 │ 0.333       │\n"
        "│ PCS23-UCS │ 0.311  │ 0.289     │ 0.325   │ 0.379   │ 0.3        │ 0.381 │ 0.332 │ 0.253       │\n"
        "│ CIELAB    │ 0.426  │ 0.41      │ 0.433   │ 0.543   │ 0.334      │ 0.517 │ 0.401 │ 0.308       │\n"
        "│ ProLab    │ 0.441  │ 0.451     │ 0.429   │ 0.485   │ 0.302      │ 0.519 │ 0.394 │ 0.289       │\n"
        "│ Oklab     │ 0.471  │ 0.515     │ 0.424   │ 0.416   │ 0.318      │ 0.452 │ 0.45  │ 0.132       │\n"
        "│ JzAzBz    │ 0.418  │ 0.404     │ 0.424   │ 0.494   │ 0.385      │ 0.474 │ 0.451 │ 0.244       │\n"
        "│ ICaCb     │ 0.391  │ 0.396     │ 0.38    │ 0.424   │ 0.248      │ 0.474 │ 0.373 │ 0.192       │\n"
        "│ ICtCp     │ 0.463  │ 0.481     │ 0.441   │ 0.636   │ 0.423      │ 0.559 │ 0.397 │ 0.34        │\n"
        "└───────────┴────────┴───────────┴─────────┴─────────┴────────────┴───────┴───────┴─────────────┘\n"
        "Legend: Best Second Best\n"
    )

    def test_output(self):
        from vsl_ial.eval.eval import main
        from rich.console import Console
        from io import StringIO

        file = StringIO()
        console = Console(record=True, file=file, width=120)
        main(console)
        self.assertEqual(console.export_text(), self.REF)
