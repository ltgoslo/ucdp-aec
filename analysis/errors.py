import argparse
import datetime
import json
import pathlib
import sys
from typing import Any, Dict, Set, Optional

import datasets


def load_predictions(prediction_path: pathlib.Path) -> Dict[int, Dict[str, Any]]:
    predictions = {}
    with prediction_path.open("r") as f:
        for line_number, line in enumerate(f):
            obj = json.loads(line)
            if "id" not in obj:
                sys.exit(1)
            predictions[obj["id"]] = obj
    return predictions


def load_dataset(split_path: pathlib.Path) -> Dict[str, datasets.Dataset]:
    splits = {
        split: datasets.load_from_disk(str(split_path / ".." / split))
        for split in ["train", "validation", "test"]
    }
    return splits


def extract_actor_set(dataset: datasets.Dataset) -> Set[str]:
    actors = set()
    for sample in dataset:
        if "side_a_name" in sample and sample["side_a_name"]:
            actors.add(sample["side_a_name"])
        if "side_b_name" in sample and sample["side_b_name"]:
            actors.add(sample["side_b_name"])
    return actors


def evaluate_actor_accuracy_on_unseen(
        predictions: Dict[int, Dict[str, Any]],
        references: datasets.Dataset,
        train_actors: Set[str],
        key: str,
    ) -> float:
    total = 0
    correct = 0
    for ref in references:
        if key not in ref or ref["id"] not in predictions:
            continue
        gold = ref[key]
        pred = predictions[ref["id"]].get(key)
        if gold not in train_actors:
            total += 1
            if gold == pred:
                correct += 1
    print(f"Proportion of unseen actors in {key} predictions: {total/len(references)*100:.2f}%")
    return correct / total if total > 0 else float("nan")

def parse_date(date_str: str) -> Optional[datetime.date]:
    try:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def evaluate_date_accuracy_on_nonzero_duration(
    predictions: Dict[int, Dict[str, Any]],
    references: datasets.Dataset,
) -> Dict[str, float]:
    total = 0
    correct_start = 0
    correct_end = 0

    for ref in references:
        if ref["id"] not in predictions:
            continue

        gold_start = ref.get("start_date", "")
        gold_end = ref.get("end_date", "")
        if not gold_start or not gold_end:
            continue

        if (gold_end - gold_start).days == 0:
            continue  # Skip zero-duration events

        pred = predictions[ref["id"]]
        pred_start = parse_date(pred.get("start_date", ""))
        pred_end = parse_date(pred.get("end_date", ""))
        if not pred_start or not pred_end:
            continue

        total += 1
        if gold_start == pred_start:
            correct_start += 1
        if gold_end == pred_end:
            correct_end += 1

    return {
        "start_date_acc_nw": round(correct_start / total, 3) if total > 0 else float("nan"),
        "end_date_acc_nw": round(correct_end / total, 3) if total > 0 else float("nan"),
    }


def evaluate_date_accuracy_on_nonzero_reporting(
    predictions: Dict[int, Dict[str, Any]],
    references: datasets.Dataset,
) -> Dict[str, float]:
    total = 0
    correct_start = 0
    correct_end = 0

    for ref in references:
        if ref["id"] not in predictions:
            continue

        gold_start = ref.get("start_date", "")
        gold_source = ref.get("source_date", "").date()
        if not gold_start or not gold_source:
            continue

        if (gold_source - gold_start).days == 0:
            continue  # Skip zero-reporting delay events

        pred = predictions[ref["id"]]
        pred_start = parse_date(pred.get("start_date", ""))
        pred_end = parse_date(pred.get("end_date", ""))
        if not pred_start or not pred_end:
            continue

        total += 1
        if gold_start == pred_start:
            correct_start += 1
        if gold_source == pred_end:
            correct_end += 1

    return {
        "start_date_acc_nr": round(correct_start / total, 3) if total > 0 else float("nan"),
        "end_date_acc_nr": round(correct_end / total, 3) if total > 0 else float("nan"),
    }


def main(split_path: pathlib.Path, prediction_path: pathlib.Path) -> None:
    predictions = load_predictions(prediction_path)
    splits = load_dataset(split_path)

    test_split = splits[split_path.stem]
    train_split = splits.get("train", [])

    # Actor analysis
    seen_actors = extract_actor_set(train_split)

    acc_a = evaluate_actor_accuracy_on_unseen(predictions, test_split, seen_actors, "side_a_name")
    acc_b = evaluate_actor_accuracy_on_unseen(predictions, test_split, seen_actors, "side_b_name")

    print(f"Actor accuracy for side_a_name (unseen in training): {acc_a:.3f}")
    print(f"Actor accuracy for side_b_name (unseen in training): {acc_b:.3f}")

    # Date analysis
    date_acc_nw = evaluate_date_accuracy_on_nonzero_duration(predictions, test_split)
    print(f"Start/end date accuracies (event estimation window ≠ 0): {date_acc_nw}")

    date_accs_nr = evaluate_date_accuracy_on_nonzero_reporting(predictions, test_split)
    print(f"Start/end date accuracies (event reporting delay ≠ 0): {date_accs_nr}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=pathlib.Path, help="Path to the target dataset split.")
    parser.add_argument("prediction", type=pathlib.Path, help="Path to the jsonl prediction file.")
    args = parser.parse_args()

    main(args.split, args.prediction)
