import argparse
import collections
import pathlib
import re

import datasets


NUM_COUNTRIES = 12


def main(dataset_path: pathlib.Path) -> None:
    split_size: dict[str, int] = {}
    per_split: dict[str, dict[str, int]] = {}
    for split_path in dataset_path.iterdir():
        if split_path.is_dir():
            split = datasets.load_from_disk(str(split_path))
            split_size[split_path.name] = len(split)
            per_split[split_path.name] = collections.Counter(split["location_root_name"])

    max_rate_per_country: dict[str, float] = collections.Counter()
    for split, countries in per_split.items():
        for country, cardinal in countries.items():
            rate: float = cardinal / split_size[split]
            max_rate_per_country[country] = max(max_rate_per_country[country], rate)

    selection: list[str] = [ country for country, _ in max_rate_per_country.most_common(NUM_COUNTRIES) ]
    selection.sort(key=lambda country: per_split["train"][country], reverse=True)

    for index, country in enumerate(selection):
        short_country: str = re.sub(r" \(.*", "", country)
        rates: dict[str, float] = { split: per_split[split][country] / split_size[split] for split in split_size }
        print(r"\countryDistrib"\
                f"{{{index}}}"\
                f"{{{short_country}}}"\
                f"{{{rates['train']}}}"\
                f"{{{rates['validation']}}}"\
                f"{{{rates['test']}}}"\
                r"\\")


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Generate country statistics for TikZ figure.")
    parser.add_argument("dataset", type=pathlib.Path, help="Path to the target dataset splits.")

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(dataset_path=args.dataset)
