from typing import Any
import argparse
import collections
import json
import pathlib
import zipfile

import datasets

from hplt_align.common import map_minhash


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Update a huggingface dataset using a minhash input and output.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output dataset splits.")
    parser.add_argument("dataset", type=pathlib.Path, help="Path to the input dataset splits.")
    parser.add_argument("sources", type=pathlib.Path, help="Path to the source jsonl input file.")
    parser.add_argument("minhash", type=pathlib.Path, help="Path to the output zip of the minhash script.")

    args: argparse.Namespace = parser.parse_args()
    return args


def read_sources(sources_path: pathlib.Path) -> dict[int, int]:
    """Returns a dictionary mapping UCDP ids to line ids."""
    sources: dict[int, int] = {}
    with sources_path.open("r") as sources_file:
        for line_number, source_line in enumerate(sources_file):
            source: dict[str, Any] = json.loads(source_line)
            sources[source["id"]] = line_number + 1  # Line numbers start at 1
    return sources


def match_filter(match: dict[str, Any]) -> bool:
    # Filter out non-English matches and short texts and long texts
    return match["qid"].startswith("deduplicated/eng_Latn/") and 100 <= len(match["text"]) <= 10000


def read_minhash(minhash_path: pathlib.Path) -> dict[int, list[dict[str, Any]]]:
    """Returns a dictionary mapping each line id to a list of HPLT match in order of approximate Jacard similarity."""
    minhash: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    map_minhash(minhash_path, lambda match: minhash[abs(match["tid"])].append(match))

    for tid, matches in minhash.items():
        minhash[tid] = sorted(list(filter(match_filter, matches)), key=lambda match: match["sim"], reverse=True)
    return minhash


def process_split(output_path: pathlib.Path, split_path: pathlib.Path, sources: dict[int, int], minhash: dict[int, list[dict[str, Any]]], source_count: dict[str, int]) -> None:
    datasets.load_from_disk(str(split_path))\
            .filter(input_columns=["id"], load_from_cache_file=False,
                    function = lambda eid: eid in sources and minhash[sources[eid]] and source_count[minhash[sources[eid]][0]["text"]]==1)\
            .map(input_columns=["id"], load_from_cache_file=False,
                 function = lambda eid: {"source_article": minhash[sources[eid]][0]["qid"]})\
            .save_to_disk(str(output_path))


def main(output_path: pathlib.Path, dataset_path: pathlib.Path, sources_path: pathlib.Path, minhash_path: pathlib.Path) -> None:
    output_path.mkdir(parents=True)
    sources: dict[int, int] = read_sources(sources_path)
    minhash: dict[int, list[dict[str, Any]]] = read_minhash(minhash_path)

    # If two documents are mapped to the same HPLT document, it indicates a document of poor quality, count them in order to remove repeated ones.
    source_count: dict[str, int] = collections.Counter(
            minhash[sources[sample["id"]]][0]["text"]
            for split_path in dataset_path.iterdir()
            for sample in datasets.load_from_disk(str(split_path))
            if sample["id"] in sources and minhash[sources[sample["id"]]])
    for split_path in dataset_path.iterdir():
        process_split(output_path / split_path.name, split_path, sources, minhash, source_count)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, dataset_path=args.dataset, sources_path=args.sources, minhash_path=args.minhash)
