from typing import Any, Callable
import json
import pathlib
import tarfile
import zipfile

import tqdm


def map_minhash(minhash_path: pathlib.Path, function: Callable[[dict[str, Any]], None]) -> None:
    if minhash_path.name.endswith('.zip'):
        return map_minhash_zip(minhash_path, function)
    if minhash_path.name.endswith('.tar.gz') or minhash_path.name.endswith('.tgz') :
        return map_minhash_tgz(minhash_path, function)
    raise NotImplemented("Minhash archive format.")


def map_minhash_tgz(minhash_path: pathlib.Path, function: Callable[[dict[str, Any]], None]) -> None:
    with tarfile.open(minhash_path, "r:gz") as tar:
        for fileobject in tqdm.tqdm(tar.getmembers(), desc="Loading minhash output"):
            if not fileobject.name.endswith(".out"):
                continue
            file = tar.extractfile(fileobject)
            if file is not None:
                for line in file:
                    function(json.loads(line))


def map_minhash_zip(minhash_path: pathlib.Path, function: Callable[[dict[str, Any]], None]) -> None:
    with zipfile.ZipFile(minhash_path) as zfile:
        for filename in tqdm.tqdm(zfile.namelist(), desc="Loading minhash output"):
            if not filename.endswith(".out"):
                continue
            with zfile.open(filename, "r") as file:
                for line in file:
                    function(json.loads(line))
