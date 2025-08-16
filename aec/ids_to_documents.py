from typing import Any, Iterator
import argparse
import collections
import json
import os
import pathlib
import requests
import sys
import urllib.request

import tqdm


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


def hplt_stream_lines(file: str) -> Iterator[str]:
    import zstandard
    response = requests.get(f"{HPLT_URL_PREFIX}/{file}", stream=True)
    response.raise_for_status()
    file_size = int(response.headers.get("content-length", 0))
    with tqdm.tqdm(desc=file, total=file_size, unit="B", unit_scale=True) as progress_bar:
        decompressor = zstandard.ZstdDecompressor().decompressobj()
        data_buffer: bytes = b""
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            progress_bar.update(len(chunk))
            data_buffer += decompressor.decompress(chunk)
            if data_buffer:
                lines: list[bytes] = data_buffer.split(b"\n")
                data_buffer = lines[-1]
                yield from map(bytes.decode, lines[:-1])
        if data_buffer:
            yield data_buffer


def hplt_cacheless_download_lines(file: str, line_numbers: list[int]) -> dict[str, str]:
    data: dict[str, str] = {}
    lines_to_select: set[int] = set(line_numbers)
    line_number: int
    line: str
    for line_number, line in enumerate(hplt_stream_lines(file), start=1):
        if line_number in lines_to_select:
            document: dict[str, str] = json.loads(line)
            data[f"{file}:{line_number}"] = document["text"]
            lines_to_select.remove(line_number)
            if not lines_to_select:
                break
    if lines_to_select:
        raise RuntimeError(f"Invalid document id {file}:{lines_to_select}")
    return data


def hplt_cacheless_data(hplt_ids: list[str]) -> dict[str, str]:
    files: dict[str, list[int]] = collections.defaultdict(list)
    for hplt_id in hplt_ids:
        file, line_number = hplt_id.split(":")
        files[file].append(int(line_number))
    data: dict[str, str] = {}
    for file, line_numbers in files.items():
        data.update(hplt_cacheless_download_lines(file, line_numbers))
    return data


def hplt_cached_data(hplt_ids: list[str]) -> dict[str, str]:
    try:
        cache: dict[str, str] = {}
        with urllib.request.urlopen(CACHE_URL) as cache_file:
            for line in cache_file:
                cached_document: dict[str, str] = json.loads(line)
                cache[cached_document["id"]] = cached_document["text"]
        return cache
    except:
        print("\n===========================================================================================\nERROR: You need to update the UCDP-AEC repository, the current version has been deprecated.\n===========================================================================================", file=sys.stderr)
        raise


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
    hplt_data: dict[str, str] = hplt_cacheless_data(hplt_ids) if cacheless else hplt_cached_data(hplt_ids)
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
