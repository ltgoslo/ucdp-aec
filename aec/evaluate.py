from __future__ import annotations

from typing import Any, ClassVar, cast
import argparse
import datetime
import itertools
import json
import logging
import pathlib
import sys

import datasets
import tqdm
import torch
import transformers


logger: logging.Logger = logging.getLogger(__name__)


FIELDS: set[str] = {
    "side_a_name", "side_b_name",  # Actors
    "start_date", "end_date",  # Time
    "location_root_name", "location_adm1_name", "location_adm2_name", "location_where_name",  # Location
    "deaths_side_a", "deaths_side_b", "deaths_civilian", "deaths_unknown", "deaths_low", "deaths_high",  # Deaths
}


SEMANTIC_MODEL: str = "google-bert/bert-base-uncased" # "sentence-transformers/all-mpnet-base-v2"


class Metrics:
    ACCURACY_FIELDS: set[str] = FIELDS

    RMSE_FIELDS: set[str] = {
        "start_date", "end_date",
        "deaths_side_a", "deaths_side_b", "deaths_civilian", "deaths_unknown", "deaths_low", "deaths_high",
    }

    def __init__(self: Metrics, dataset: datasets.Dataset | None) -> None:
        self.sample_count: int = 0
        self.correct: dict[str, int] = { field: 0 for field in self.ACCURACY_FIELDS }
        self.square_error: dict[str, int] = { field: 0 for field in self.RMSE_FIELDS }
        self.semantic_evaluation: bool = dataset is not None
        if self.semantic_evaluation:
            self.cosine: dict[str, float] = { field: 0 for field in self.ACCURACY_FIELDS }
            self.reciprocal_rank: dict[str, float] = { field: 0 for field in self.ACCURACY_FIELDS }
            self.precision1: dict[str, int] = { field: 0 for field in self.ACCURACY_FIELDS }
            self._semantic_init(dataset)

    def _semantic_init(self: Metrics, dataset: datasets.Dataset) -> None:
        self._semantic_tokenizer = transformers.AutoTokenizer.from_pretrained(SEMANTIC_MODEL)
        self._semantic_model = transformers.AutoModel.from_pretrained(SEMANTIC_MODEL)
        self._candidates: dict[str, list[str]] = { field: list(map(str, set(dataset[field]))) for field in self.ACCURACY_FIELDS }
        self._embedding_cache: dict[str, torch.Tensor] = {}
        all_values: set[str] = set(itertools.chain(*self._candidates.values()))
        self._embedding_cache = {key: self._semantic_embedding(key) for key in tqdm.tqdm(all_values, desc="Embedding gold candidates")}
        self._candidates_embeddings: dict[str, torch.Tensor] = {field: torch.stack([self._embedding_cache[key] for key in candidate_values]) for field, candidate_values in self._candidates.items()}

    def _semantic_embedding(self: Metrics, value: str) -> torch.Tensor:
        if value in self._embedding_cache:
            return self._embedding_cache[value]
        with torch.no_grad():
            input_tensors: dict[str, torch.Tensor] = self._semantic_tokenizer(value, return_tensors="pt")
            cls_embedding: torch.Tensor = self._semantic_model(**input_tensors).last_hidden_state[0,0]
            return cls_embedding

    def _update_semantic_evaluation(self: Metrics, field: str, gold: str, prediction: str | None) -> None:
        if prediction is None:
            self.cosine[field] += -1
            return # MRR is 0

        gold_index: int = self._candidates[field].index(gold)
        prediction_embedding: torch.Tensor = self._semantic_embedding(prediction)
        similarities: torch.Tensor = torch.cosine_similarity(self._candidates_embeddings[field], prediction_embedding)
        self.cosine[field] += cast(float, similarities[gold_index].item())
        ranks: list[int] = cast(list[int], torch.argsort(similarities, descending=True).tolist())
        gold_rank: int = ranks.index(gold_index)
        self.reciprocal_rank[field] += 1/(1+gold_rank)
        self.precision1[field] += (gold_rank == 0)

    def update(self: Metrics, gold: dict[str, Any], prediction: dict[str, Any]) -> None:
        self.sample_count += 1

        missing_fields: set[str] = FIELDS - prediction.keys()
        if missing_fields:
            logger.warning(f"Missing predicted field{'s' if len(missing_fields)>1 else ''} for sample id={gold['id']}: {', '.join(missing_fields)}")

        gold = {key: value.strip() if isinstance(value, str) else value for key, value in gold.items()}
        prediction = {key: value.strip() if isinstance(value, str) else value for key, value in prediction.items()}

        # The actors are permutation invariant, if inverting them would get a better accuracy
        if gold["side_a_name"] != prediction.get("side_a_name") and gold["side_b_name"] != prediction.get("side_b_name") \
                and (gold["side_a_name"] == prediction.get("side_b_name") or gold["side_b_name"] == prediction.get("side_a_name")):
            logger.debug(f"Switching sides for sample id={gold['id']}")
            # invert the name of the actors
            prediction["side_a_name"], prediction["side_b_name"] = prediction.get("side_b_name"), prediction.get("side_a_name")
            # invert the number of deaths on each side
            prediction["deaths_side_a"], prediction["deaths_side_b"] = prediction.get("deaths_side_b"), prediction.get("deaths_side_a")

        for field in self.ACCURACY_FIELDS:
            self.correct[field] += (str(prediction.get(field, "")) == str(gold[field]))
            if self.semantic_evaluation:
                self._update_semantic_evaluation(field, str(gold[field]), str(prediction[field]) if field in prediction else None)

        for field in self.RMSE_FIELDS:
            # The field might have been set to None by the actor inversion logic
            prediction_value = prediction.get(field)
            if field.startswith("deaths_"):
                if prediction_value is None:
                    prediction_value = 0
                difference = (gold[field] - prediction_value)**2
            elif field.endswith("_date"):
                if isinstance(prediction_value, str):
                    try:
                        prediction_value = datetime.date.fromisoformat(prediction_value)
                    except ValueError:
                        logger.info(f"Couldn't parse {field} as date for sample id={gold['id']}")
                        prediction_value = None
                if prediction_value is None:
                    prediction_value = gold["source_date"].date()
                difference = (gold[field] - prediction_value).days**2
            self.square_error[field] += difference

    def report(self: Metrics) -> None:
        measures: list[str] = ["accuracy"]
        atoms: dict[str, float] = {}
        for field in self.ACCURACY_FIELDS:
            atoms[f"{field} accuracy"] = 100 * self.correct[field] / self.sample_count
        for field in self.RMSE_FIELDS:
            atoms[f"{field} RMSE"] = (self.square_error[field] / self.sample_count)**0.5
        if self.semantic_evaluation:
            measures.extend(["semantic MRR", "semantic p@1", "semantic cos"])
            for field in self.ACCURACY_FIELDS:
                atoms[f"{field} semantic MRR"] = self.reciprocal_rank[field] / self.sample_count
                atoms[f"{field} semantic p@1"] = self.precision1[field] / self.sample_count
                atoms[f"{field} semantic cos"] = self.cosine[field] / self.sample_count
        
        aggregates: dict[str, float] = {}
        for measure in measures:
            aggregates[f"actor {measure}"] = (atoms[f"side_a_name {measure}"] + atoms[f"side_b_name {measure}"]) / 2
            aggregates[f"date {measure}"] = (atoms[f"start_date {measure}"] + atoms[f"end_date {measure}"]) / 2
            aggregates[f"location {measure}"] = (atoms[f"location_root_name {measure}"] + atoms[f"location_adm1_name {measure}"] + atoms[f"location_adm2_name {measure}"] + atoms[f"location_where_name {measure}"]) / 4
            aggregates[f"deaths {measure}"] = (atoms[f"deaths_side_a {measure}"] + atoms[f"deaths_side_b {measure}"] + atoms[f"deaths_civilian {measure}"] + atoms[f"deaths_unknown {measure}"] + atoms[f"deaths_low {measure}"] + atoms[f"deaths_high {measure}"]) / 6
            aggregates[f"aggregate {measure}"] = (aggregates[f"actor {measure}"] + aggregates[f"date {measure}"] + aggregates[f"location {measure}"] + aggregates[f"deaths {measure}"]) / 4
        aggregates["date RMSE"] = (atoms["start_date RMSE"] + atoms["end_date RMSE"]) / 2
        aggregates["deaths RMSE"] = (atoms["deaths_side_a RMSE"] + atoms["deaths_side_b RMSE"] + atoms["deaths_civilian RMSE"] + atoms["deaths_unknown RMSE"] + atoms["deaths_low RMSE"] + atoms["deaths_high RMSE"]) / 6

        print(json.dumps(atoms | aggregates))
        for result in [atoms, aggregates]:
            print(38*"=")
            for field, value in result.items():
                if field.endswith("accuracy"):
                    print(f"{field:<32} | {value:7.1f}")
                else:
                    print(f"{field:<32} | {value:7.4f}")


def parse_command_argument() -> argparse.Namespace:
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(description="Update a huggingface dataset using a minhash input and output.")
    parser.add_argument("split", type=pathlib.Path, help="Path to the target dataset split.")
    parser.add_argument("prediction", type=pathlib.Path, help="Path to the jsonl prediction file.")
    parser.add_argument("-d", "--debug", action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING, help="Print lots of debugging statements.")
    parser.add_argument("-s", "--no-semantic", action="store_true", help="Do not run semantic evaluation.")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.INFO, help="Be verbose.")
    args: argparse.Namespace = parser.parse_args()
    return args


def main(split_path: pathlib.Path, prediction_path: pathlib.Path, semantic: bool) -> None:
    predictions: dict[int, dict[str, Any]] = {}
    with prediction_path.open("r") as prediction_file:
        for line_number, prediction_line in enumerate(prediction_file):
            prediction: dict[str, Any] = json.loads(prediction_line)
            if "id" not in prediction:
                logger.fatal(f"Prediction file contains an object without id line {line_number+1}.")
                sys.exit(1)
            predictions[prediction["id"]] = prediction

    dataset: datasets.Dataset
    if semantic:
        splits: dict[str, datasets.Dataset] = {split: datasets.load_from_disk(str(split_path / ".." / split))
                                               for split in ["train", "validation", "test"]}
        dataset = splits[split_path.stem]
        metrics = Metrics(datasets.concatenate_datasets(splits.values()))
    else:
        dataset = datasets.load_from_disk(str(split_path))
        metrics = Metrics(None)
    for sample in tqdm.tqdm(dataset, desc="Scoring samples"):
        if sample["id"] not in predictions:
            logger.warning(f"Missing prediction for sample id={sample['id']}")
        metrics.update(sample, predictions.get(sample["id"], {}))
    metrics.report()


if __name__ == "__main__":
    args: argparse.Namespace = parse_command_argument()
    logging.basicConfig(level=args.loglevel)
    main(split_path=args.split, prediction_path=args.prediction, semantic=not args.no_semantic)
