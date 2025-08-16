"""Module for batching socio-political datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import transformers


if TYPE_CHECKING:
    import t5.dataset
    import t5.model


class DataCollator:
    """Group socio-political event samples into a batch."""

    # We need a custom collator in order to keep all features in the batch, including some that cannot be batched automatically.
    def __init__(self: DataCollator, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
        """Initialize socio-political event data collator."""
        self.dataset = dataset
        self.model = model
        self.text_collator = transformers.data.DataCollatorForSeq2Seq(tokenizer=self.dataset.tokenizer, model=self.model)
        self.text_keys: list[str] = ["input_ids", "attention_mask", "labels"]

    def __call__(self: DataCollator, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a list of samples into a batch."""
        text_inputs: list[dict[str, Any]] = [{key: sample[key] for key in self.text_keys} for sample in samples]
        batch: dict[str, Any] = {key: [] for key in samples[0] if key not in self.text_keys}
        batch.update(self.text_collator(text_inputs))
        for sample in samples:
            for key, value in sample.items():
                if key not in self.text_keys:
                    batch[key].append(value)
        return batch
