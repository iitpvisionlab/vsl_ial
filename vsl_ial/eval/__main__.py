from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import json5
from rich.console import Console

FILE = Path(__file__)
DISTANCE_PATH = FILE.with_name("distance-pcs23-ucs-article.json")
HUE_LINEARITY_PATH = FILE.with_name("hue-linearity-pcs23-ucs-article.json")


def distance_command(
    config_path: Path, update_schema: bool, console: Console | None = None
):
    from .eval import main, Config

    if update_schema:
        schema_path = FILE.with_name("distance_schema.json")
        schema_path.write_text(
            json5.dumps(
                Config.model_json_schema(), ensure_ascii=False, quote_keys=True
            )
        )
        print(f"{schema_path} updated")
        return

    config = Config(**json5.loads(config_path.read_text()))  # type: ignore

    main(config, console)


def hue_linearity_command(
    config_path: Path, update_schema: bool, console: Console | None = None
):
    from .eval2 import main, Config

    if update_schema:
        schema_path = FILE.with_name("hue_linearity_schema.json")
        schema_path.write_text(
            json5.dumps(
                Config.model_json_schema(), ensure_ascii=False, quote_keys=True
            )
        )
        print(f"{schema_path} updated")
        return

    config = Config(**json5.loads(config_path.read_text()))  # type: ignore

    main(config, console)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    distance_parser = subparsers.add_parser("distance")
    distance_parser.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=DISTANCE_PATH,
        help="default: %(default)s",
    )
    distance_parser.set_defaults(func=distance_command)

    hue_linearity = subparsers.add_parser("hue-linearity")
    hue_linearity.add_argument(
        "--config",
        dest="config_path",
        type=Path,
        default=HUE_LINEARITY_PATH,
        help="default: %(default)s",
    )
    hue_linearity.set_defaults(func=hue_linearity_command)

    for subparser in distance_parser, hue_linearity:
        subparser.add_argument(
            "--update-schema",
            action="store_true",
            help="generate json schema file so that VSCode can check syntax",
        )

    args = vars(parser.parse_args())
    func = args.pop("func")
    func(**args)

if __name__ == "__main__":  # called from tests
    main()
