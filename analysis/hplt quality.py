from typing import Any
import argparse
import collections
import pathlib
import json


DOMAINS: list[str] = ["actor", "date", "location", "deaths"]


def main() -> None:
    annotations_path: pathlib.Path = pathlib.Path(__file__).parent / ".." / "hplt_align" / "align_annotations.jsonl"
    total: int = 0
    similars: int = 0
    partial_error: int = 0
    counts: dict[str, int] = { domain: 0 for domain in DOMAINS }
    with annotations_path.open("r") as annotations_file:
        for line in annotations_file:
            annotation: dict[str, Any] = json.loads(line)
            total += 1
            if annotation["similar"]:
                similars += 1
                perfect = True
                for domain in DOMAINS:
                    if annotation[f"wrong_{domain}"]:
                        counts[domain] += 1
                        perfect = False
                if not perfect:
                    partial_error += 1
            else:
                partial_error += 1

    perfect: int = total - partial_error
    completly_wrong: int = total - similars
    print(f"Total documents annotated: {total}")
    print(f"Perfects:        {perfect:4} / {100*perfect/total:5.1f}%")
    print(f"Similars:        {similars:4} / {100*similars/total:5.1f}%")
    print(f"Partial error:   {partial_error:4} / {100*partial_error/total:5.1f}%")
    print(f"Completly wrong: {completly_wrong:4} / {100*completly_wrong/total:5.1f}%")
    for domain, count in counts.items():
        print(f"Domain {domain:8}: {count:4} / {100*count/total:5.1f}%")


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Generate statistics from HPLT align annotations.")
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main()
