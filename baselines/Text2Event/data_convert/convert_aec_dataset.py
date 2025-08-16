from typing import Any
import argparse
import json
import pathlib

import datasets

from ucdp_fields import UCDP_FIELDS


SPLIT_NAME_MAP: dict[str, str] = {
    "train": "train",
    "validation": "val",
    "test": "test"
}


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Convert a AEC dataset into a Text2Event dataset.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output Text2Event dataset directory.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the input AEC dataset.")

    args: argparse.Namespace = parser.parse_args()
    return args


def write_schema(output_path: pathlib.Path) -> None:
    with output_path.open("w") as output_file:
        print(json.dumps(["UCDP"]), file=output_file)
        print(json.dumps([label for _, label, _ in UCDP_FIELDS]), file=output_file)
        print(json.dumps({"UCDP": [label for _, label, _ in UCDP_FIELDS]}), file=output_file)


def input_template(sample: dict[str, Any]) -> str:
    return f"{sample['source_date']} {sample['source_article']}"


def output_template(sample: dict[str, Any]) -> str:
    return "<extra_id_0> " + " ".join(f"<extra_id_0> {label} {sample[field]} <extra_id_1>" for field, label, _ in UCDP_FIELDS) + " <extra_id_1>"


def convert_split(output_path: pathlib.Path, split: datasets.Dataset) -> None:
    with output_path.open("w") as output_file:
        for sample in split:
            print(json.dumps({
                    "text": input_template(sample),
                    "event": output_template(sample),
                }), file=output_file)


def main(output_path: pathlib.Path, input_path: pathlib.Path):
    output_path.mkdir(parents=True, exist_ok=True)
    write_schema(output_path / "event.schema")
    for path in input_path.iterdir():
        if path.suffix == "":
            convert_split(output_path / f"{SPLIT_NAME_MAP[path.name]}.json", datasets.load_from_disk(str(path)))


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, input_path=args.input)
