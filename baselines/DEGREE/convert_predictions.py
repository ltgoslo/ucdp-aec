import argparse
import contextlib
import itertools
import json
import pathlib

import datasets


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Convert a DEGREE predictions file into an AEC predictions file.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output AEC predictions.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the input DEGREE predictions.")
    parser.add_argument("split", type=pathlib.Path, help="Path to the AEC split that resulted in the given predictions.")

    args: argparse.Namespace = parser.parse_args()
    return args


def main(output_path: pathlib.Path, input_path: pathlib.Path, split_path: pathlib.Path) -> None:
    split: datasets.Dataset = datasets.load_from_disk(str(split_path))
    with input_path.open("r") as input_file:
        degree_predictions = json.load(input_file)

    with output_path.open("w") as output_file:
        for sample_id, degree_object in zip(split["id"], degree_predictions):
            arguments: dict[str, str] = {"id": sample_id}
            for role, argument in degree_object["pred arguments"].items():
                argument = argument.strip()
                if role.startswith("deaths_"):
                    with contextlib.suppress(ValueError):
                        arguments[role] = int("".join(itertools.takewhile(str.isdigit, argument)))
                else:
                    arguments[role] = argument
            print(json.dumps(arguments), file=output_file)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, input_path=args.input, split_path=args.split)
