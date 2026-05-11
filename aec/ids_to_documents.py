from typing import Any, Iterator
import argparse
import bisect
import collections
import io
import json
import operator
import pathlib
import struct
import sys
import urllib.request

import tqdm
import zstandard


CACHE_URL: str = "https://recurrent.network/AEC/2025.cache"
HPLT_URL_PREFIX: str = "https://data.hplt-project.org/two"
DOWNLOAD_CHUNK_SIZE: int = 8192


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Transform a UCDP-AEC-ids dataset into UCDP-AEC by replacing HPLT ids by with source documents.")
    parser.add_argument("output", type=pathlib.Path, help="Path to the output dataset splits.")
    parser.add_argument("input", type=pathlib.Path, help="Path to the input dataset splits.")
    parser.add_argument("-C", "--no-cache", action="store_true", help="Do not use the cached subset (WARNING: requires a lot of bandwidth).")
    parser.add_argument("-J", "--jsonl-only", action="store_true", help="Only transform the jsonl data, do not process the huggingface dataset (do not require the library to be installed)")

    args: argparse.Namespace = parser.parse_args()
    return args


def process_hf_split(output_path: pathlib.Path, split_path: pathlib.Path, hplt_data: dict[str, str]) -> None:
    import datasets
    datasets.load_from_disk(str(split_path))\
            .map(input_columns=["source_article"], load_from_cache_file=False,
                 function = lambda source_article: {"source_article": hplt_data[source_article]})\
            .save_to_disk(str(output_path))


def process_jsonl_split(output_path: pathlib.Path, split_path: pathlib.Path, hplt_data: dict[str, str]) -> None:
    with output_path.open("w") as output_file, split_path.open("r") as input_file:
        for line in input_file:
            sample = json.loads(line)
            sample["source_article"] = hplt_data[sample["source_article"]]
            print(json.dumps(sample), file=output_file)


def hplt_build_index(url: str) -> list[tuple[int, int]]:
    with urllib.request.urlopen(url) as file:
        data: bytes = file.read()

    filename_length: int = data.find(b'\0')
    header_length: int = filename_length + 5
    number_of_frames: int = struct.unpack("<I", data[filename_length+1:header_length])[0]

    cumulative_sum: tuple[int, int] = (0, 0)
    index: list[tuple[int, int]] = [cumulative_sum]
    for i in range(number_of_frames):
        start: int = header_length + i*8
        newlines, compressed_size = struct.unpack("<2I", data[start:start+8])
        cumulative_sum = (cumulative_sum[0] + newlines, cumulative_sum[1] + compressed_size)
        index.append(cumulative_sum)
    return index


def hplt_get_line(url: str, index: list[tuple[str, str]], line_number: int) -> str:
    right = bisect.bisect_right(index, line_number, key=operator.itemgetter(0))
    left = right - 1
    while left > 0 and index[left][0] == line_number:
        left -= 1
    newlines_offset = index[left][0]

    start = index[left][1]
    end = index[right][1]

    with urllib.request.urlopen(
            urllib.request.Request(url,
                                   headers={"Range": f"bytes={start}-{end-1}"})) as file:
        data: bytes = file.read()
        assert(len(data) == end - start)

    decompressor = zstandard.ZstdDecompressor()
    reader = decompressor.stream_reader(io.BytesIO(data), read_across_frames=True)
    return reader.readall().split(b'\n')[line_number - newlines_offset].decode()


def hplt_download_lines(filename: str, line_numbers: list[int]) -> dict[str, str]:
    indexable_path: str = f"{HPLT_URL_PREFIX}/{filename[:-4]}.1M.zst"
    index_path: str = f"{indexable_path}.zindex"

    index: list[tuple[int, int]] = hplt_build_index(index_path)

    data: dict[str, str] = {}
    line_number: int
    for line_number in set(line_numbers):
        line: str = hplt_get_line(indexable_path, index, line_number - 1)
        sample: dict[str, Any] = json.loads(line)
        data[f"{filename}:{line_number}"] = sample["text"]
    return data


def get_hplt_data(hplt_ids: list[str]) -> dict[str, str]:
    files: dict[str, list[int]] = collections.defaultdict(list)
    for hplt_id in hplt_ids:
        file, line_number = hplt_id.split(":")
        files[file].append(int(line_number))
    data: dict[str, str] = {}
    for file, line_numbers in tqdm.tqdm(files.items(), desc="Downloading from HPLT"):
        data.update(hplt_download_lines(file, line_numbers))
    return data


def get_hplt_cache(hplt_ids: list[str]) -> dict[str, str]:
    cache: dict[str, str] = {}
    try:
        with urllib.request.urlopen(CACHE_URL) as cache_file:
            for line in cache_file:
                cached_document: dict[str, str] = json.loads(line)
                cache[cached_document["id"]] = cached_document["text"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            print("ERROR: The cache data was invalidated, try updating the repository or using the cacheless downloader (--no-cache).", file=sys.stderr)
            sys.exit(1)
        else:
            raise
    return cache


def main(output_path: pathlib.Path, input_path: pathlib.Path, cacheless: bool, jsonl_only: bool) -> None:
    if not jsonl_only:
        import datasets

    print("Reading IDs…", flush=True, end=" ")
    hplt_ids: list[str] = []
    for split_path in input_path.iterdir():
        if split_path.suffix == ".jsonl":
            with split_path.open("r") as split_file:
                for line in split_file:
                    sample: dict[str, Any] = json.loads(line)
                    hplt_ids.append(sample["source_article"])
        elif not jsonl_only and split_path.is_dir():
            split = datasets.load_from_disk(split_path)
            hplt_ids.extend(split["source_article"])
    print("done")

    output_path.mkdir(parents=True)
    print("Downloading data…", flush=True, end=" ")
    hplt_data: dict[str, str] = get_hplt_data(hplt_ids) if cacheless else get_hplt_cache(hplt_ids)
    print("done")

    print("Transforming dataset…", flush=True, end=" ")
    for split_path in input_path.iterdir():
        if split_path.is_dir() and not jsonl_only:
            process_hf_split(output_path / split_path.name, split_path, hplt_data)
        elif split_path.suffix == ".jsonl":
            process_jsonl_split(output_path / split_path.name, split_path, hplt_data)
    print("done")


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    main(output_path=args.output, input_path=args.input, cacheless=args.no_cache, jsonl_only=args.jsonl_only)
