import argparse
import datetime
import json
import pathlib

import datasets


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Convert the huggingface dataset into jsonl files in the same director.")
    parser.add_argument("dataset", type=pathlib.Path, help="Path to the input huggingface dataset.")

    args: argparse.Namespace = parser.parse_args()
    return args


def convert(x):
    if isinstance(x, int) or isinstance(x, str):
        return x
    if isinstance(x, datetime.datetime):
        return x.strftime("%Y-%m-%d %H:%M")
    if isinstance(x, datetime.date):
        return x.strftime("%Y-%m-%d")
    raise RuntimeError("Unknown type")


def main(dataset_path: pathlib.Path) -> None:
    splits: dict[str, datasets.Dataset] = {path.name: datasets.load_from_disk(str(path)) for path in dataset_path.iterdir() }
    for name, split in splits.items():
        target: pathlib.Path = dataset_path / f"{name}.jsonl"
        with target.open("w") as output:
            for sample in split:
                print(json.dumps({key: convert(value) for key, value in sample.items()}), file=output)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(dataset_path=args.dataset)
