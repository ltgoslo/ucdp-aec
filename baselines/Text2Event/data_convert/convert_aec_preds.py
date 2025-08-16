from typing import Any
import argparse
import contextlib
import json
import pathlib

import datasets

from ucdp_fields import UCDP_FIELDS


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Convert a Text2Event predictions file into an AEC predictions file.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output AEC predictions.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the input Text2Event predictions.")
    parser.add_argument("split", type=pathlib.Path, help="Path to the AEC split that resulted in the given predictions.")

    args: argparse.Namespace = parser.parse_args()
    return args


def main(output_path: pathlib.Path, input_path: pathlib.Path, split_path: pathlib.Path) -> None:
    split: datasets.Dataset = datasets.load_from_disk(str(split_path))
    with output_path.open("w") as output_file, input_path.open("r") as input_file:
        for sample_id, input_line in zip(split["id"], input_file):
            input_line = input_line.strip()
            if input_line.startswith("<extra_id_0> <extra_id_0> "):
                input_line = input_line[len("<extra_id_0> <extra_id_0> "):]
            if input_line.endswith("<extra_id_1> <extra_id_1>"):
                input_line = input_line[:-len("<extra_id_1> <extra_id_1>")]
            children: list[str] = input_line.split("<extra_id_1> <extra_id_0> ")
            arguments: dict[str, str] = {"id": sample_id}
            for child in children:
                role: str | None = None
                for key, label, _ in sorted(UCDP_FIELDS, key=lambda field: len(field[1]), reverse=True):
                    if child.startswith(label):
                        role = key
                        break
                if role is not None:
                    argument: str = child[len(label)+1:]
                    if role.startswith("deaths_"):
                        with contextlib.suppress(ValueError):
                            arguments[role] = int(argument)
                    else:
                        arguments[role] = argument
            print(json.dumps(arguments), file=output_file)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, input_path=args.input, split_path=args.split)
