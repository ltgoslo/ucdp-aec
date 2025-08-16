import argparse
import collections
import pathlib
import re

import datasets
import numpy
import matplotlib.pyplot as plt

from datetime import datetime

import numpy as np


def reporting_delay_analysis(dataset):
    print("### Analyze differences between publication dates and event start dates")

    # Compute difference in days
    diffs = [(sample["source_date"].date() - sample["start_date"]).days for sample in dataset]
    diffs_array = np.array(diffs)

    avg_diff = diffs_array.mean()
    print(f"Average source_date - start_date difference: {avg_diff:.2f} days")

    buckets = {
        "* No reporting delay: 0 days": (diffs_array == 0),
        "* Reporting delay of up to 1 day": (diffs_array == 1),
        "* Reporting delay of up to 7 days": (diffs_array == 7),
        "* Reporting delay of up to 30 days": (diffs_array == 30),
        "* Reporting delay of more than 30 days": (diffs_array > 30)
    }

    total = len(diffs_array)
    for label, mask in buckets.items():
        percent = mask.mean() * 100
        print(f"{label}: {percent:.1f}%")

    # Plot (remove if annoying)
    bucket_counts = [mask.sum() for mask in buckets.values()]
    labels = list(buckets.keys())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, bucket_counts, color='skyblue')
    plt.ylabel("Count")
    plt.title("Distribution of (source_date - start_date) in days")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def main(dataset_path: pathlib.Path) -> None:
    splits: dict[str, datasets.Dataset] = {
            split_path.name: datasets.load_from_disk(str(split_path))
            for split_path in dataset_path.iterdir()
            if split_path.is_dir()
        }
    full = datasets.concatenate_datasets(splits.values())

    # Actors
    one_sided = sum(sample["side_b_name"] == "Civilians" for sample in full) / len(full)
    print(f"Percent of one-sided (B=civilians): {one_sided*100:.1f}%")

    # Dates
    tenyears = sum(sample["start_date"].year >= 2015 for sample in full) / len(full)
    print(f"Percent in last 10 years: {tenyears*100:.1f}%")
    diffdate = numpy.array([(sample["end_date"]-sample["start_date"]).days for sample in full])
    print(f"Percent of exact day: {(diffdate==0).mean()*100:.1f}%")
    print(f"Percent of non-exact day: {(diffdate!=0).mean()*100:.1f}%")
    print(f"Percent with spreak of more than a week: {(diffdate>7).mean()*100:.1f}%")
    reporting_delay_analysis(full)

    # Countries
    # TODO: check different countries in dataset

    # Locations
    num_countries = len(set(sample["location_root_name"] for sample in full))
    print(f"Number of countries in all splits: {num_countries}")

    # Deaths
    side_a = numpy.array(full["deaths_side_a"])
    side_b = numpy.array(full["deaths_side_b"])
    civilians = numpy.array(full["deaths_civilian"])
    unknown = numpy.array(full["deaths_unknown"])
    low = numpy.array(full["deaths_low"])
    high = numpy.array(full["deaths_high"])
    best = side_a + side_b + civilians + unknown
    print(f"Percent of exact deaths: {(low==high).mean()*100:.1f}%")
    print(f"Percent of uncertain (best=0): {(best==0).mean()*100:.1f}%")


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Generate some miscellaneous statistics.")
    parser.add_argument("dataset", type=pathlib.Path, help="Path to the target dataset splits.")

    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(dataset_path=args.dataset)
