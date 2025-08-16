import argparse
import json
import pathlib

import datasets


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Extract a jsonl with all sources from a huggingface dataset.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output jsonl file.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the input dataset splits.")

    args: argparse.Namespace = parser.parse_args()
    return args


def main(output_path: pathlib.Path, input_path: pathlib.Path):
    with output_path.open("w") as output_file:
        for split_path in input_path.iterdir():
            split = datasets.load_from_disk(str(split_path))
            for sample in split:
                print(json.dumps({"id": sample["id"], "article": sample["source_article"], "headline": sample["source_headline"]}), file=output_file)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, input_path=args.input)
