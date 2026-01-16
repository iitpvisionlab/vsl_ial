from __future__ import annotations
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from rich.table import Table
from rich.console import Console
import json5
from ..cs.xyz import XYZ
from ..cs import CS
from .. import FArray
from ..datasets.distance import DistanceDataset
from .dataset import DatasetConfig
from .metrics import Metrics
from .cs import CS as CSModel
from ._base import StrictModel


class Config(StrictModel):
    datasets: list[DatasetConfig]
    cs: list[CSModel]
    metrics: list[Metrics]


def evaluate(model: CS, dataset: DistanceDataset):
    """
    Evaluate model on a `DistanceDataset` measuring L2 distance
    """
    model_coordinates = model.convert(XYZ(), dataset.xyz)

    a_colors = np.empty((len(dataset.pairs), 3), dtype=np.float64)
    b_colors = np.empty_like(a_colors)
    for idx, (a_idx, b_idx) in enumerate(dataset.pairs):
        a_colors[idx] = model_coordinates[a_idx]
        b_colors[idx] = model_coordinates[b_idx]
    exp_distance = model.distance(a_colors, b_colors)
    ref_distance = dataset.dv
    return ref_distance, exp_distance


def main(console: Console | None = None):
    default_config_path = Path(__file__).with_name("pcs23-ucs-article.json")
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, default=default_config_path)
    parser.add_argument("--update-schema", action="store_true")
    args = parser.parse_args()
    if args.update_schema:
        schema_path = Path(__file__).with_name("schema.json")
        schema_path.write_text(
            json5.dumps(
                Config.model_json_schema(), ensure_ascii=False, quote_keys=True
            )
        )
        print(f"{schema_path} updated")
        return

    config = Config(**json5.loads(args.config.read_text()))

    # metrics_loaded = [metric.load() for metric in config.metrics]
    # cs_classes = [cs.load() for cs in config.cs]
    datasets_loaded = [dataset.load() for dataset in config.datasets]

    console = console or Console()
    for metric in config.metrics:
        # evaluation
        # make row for each color coordinate system
        metrics_table: list[list[float]] = []
        for cs in config.cs:
            metrics = []
            metrics_table.append(metrics)
            for dataset in datasets_loaded:
                ref_exp: list[tuple[FArray, FArray]] = []
                for sub_dataset in dataset.datasets:
                    model = cs.create_for(sub_dataset)
                    ref_exp.append(evaluate(model, sub_dataset))
                metrics.append(float(metric.load()(*zip(*ref_exp))))
            #

        # representation
        table = Table(title=f"{metric.name}")
        # column for cs name
        table.add_column("")
        # columns for dataset names
        for dataset in config.datasets:
            table.add_column(dataset.display_name)

        metrics_table_t = list(zip(*metrics_table))
        best_table = [sorted(row)[:2] for row in metrics_table_t]

        for cs, row in zip(config.cs, metrics_table, strict=True):
            table.add_row(
                cs.name,
                *[
                    # ugly long line for python 3.10 compatibility
                    f"{'[green][underline]' if metric==best[0] else ('[yellow][underline2]' if metric==best[-1] else '')}{metric:.3g}"
                    for metric, best in zip(row, best_table)
                ],
            )
        # rich.print(table)
        console.print(table)
    console.print(
        "[bold]Legend:[/bold] [green][underline]Best[reset] [yellow][underline2]Second Best"
    )
