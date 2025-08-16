import argparse
import json
import pathlib

import datasets

from hplt_align.common import map_minhash


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Update a huggingface dataset using a minhash input and output.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the cache file.")
    parser.add_argument("dataset", type=pathlib.Path, help="Path to the target dataset splits.")
    parser.add_argument("minhash", type=pathlib.Path, help="Path to the output zip of the minhash script.")

    args: argparse.Namespace = parser.parse_args()
    return args



def main(output_path: pathlib.Path, dataset_path: pathlib.Path, minhash_path: pathlib.Path) -> None:
    minhash: dict[str, str] = {}
    map_minhash(minhash_path, lambda match: minhash.update({match["qid"]: match["text"]}))

    with output_path.open("w") as output_file:
        for split_path in dataset_path.iterdir():
            split = datasets.load_from_disk(str(split_path))
            for hplt_id in split["source_article"]:
                print(json.dumps({"id": hplt_id, "text": minhash[hplt_id]}), file=output_file)


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, dataset_path=args.dataset, minhash_path=args.minhash)
