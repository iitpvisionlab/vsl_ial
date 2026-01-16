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


class TestEval(TestCase):
    REF = (
        "                                            group_stress                                            \n"
        "┏━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━┓\n"
        "┃           ┃ COMBVD ┃ BFD-P d65 ┃ BFD-P m ┃ BFD-P c ┃ RIT-DuPont ┃ Witt  ┃ Leeds ┃ Munsell-3.1.0* ┃\n"
        "┡━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━┩\n"
        "│ CAM16-SCD │ 0.319  │ 0.298     │ 0.358   │ 0.285   │ 0.213      │ 0.29  │ 0.251 │ 0.0808         │\n"
        "│ CAM16-UCS │ 0.305  │ 0.271     │ 0.35    │ 0.297   │ 0.206      │ 0.31  │ 0.245 │ 0.0883         │\n"
        "│ CAM16-LCD │ 0.337  │ 0.293     │ 0.378   │ 0.376   │ 0.251      │ 0.397 │ 0.303 │ 0.136          │\n"
        "│ CAM02-SCD │ 0.321  │ 0.306     │ 0.353   │ 0.303   │ 0.219      │ 0.286 │ 0.253 │ 0.0956         │\n"
        "│ CAM02-UCS │ 0.306  │ 0.28      │ 0.343   │ 0.321   │ 0.213      │ 0.305 │ 0.246 │ 0.116          │\n"
        "│ CAM02-LCD │ 0.337  │ 0.302     │ 0.37    │ 0.419   │ 0.261      │ 0.392 │ 0.309 │ 0.219          │\n"
        "│ PCS23-UCS │ 0.311  │ 0.289     │ 0.325   │ 0.379   │ 0.3        │ 0.381 │ 0.332 │ 0.0741         │\n"
        "│ CIELAB    │ 0.426  │ 0.41      │ 0.433   │ 0.543   │ 0.334      │ 0.517 │ 0.401 │ 0.281          │\n"
        "│ ProLab    │ 0.441  │ 0.451     │ 0.429   │ 0.485   │ 0.302      │ 0.519 │ 0.394 │ 0.177          │\n"
        "│ Oklab     │ 0.471  │ 0.515     │ 0.424   │ 0.416   │ 0.318      │ 0.452 │ 0.45  │ 0.074          │\n"
        "│ JzAzBz    │ 0.418  │ 0.404     │ 0.424   │ 0.494   │ 0.385      │ 0.474 │ 0.451 │ 0.127          │\n"
        "└───────────┴────────┴───────────┴─────────┴─────────┴────────────┴───────┴───────┴────────────────┘\n"
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
