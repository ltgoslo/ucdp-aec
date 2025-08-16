"""Module for handling socio-political datasets."""

from __future__ import annotations

import contextlib
import datetime
import locale
import re
from typing import TYPE_CHECKING, Any

import datasets
import transformers


if TYPE_CHECKING:
    import t5.config


class Dataset:
    """
    A socio-political event dataset.

    Config
    ------
    name: str
        Name of the dataset to train on. It must be a directory in $DATA_PATH.
    processing: str, optional
        How to process the data, must be one of "none", "generative".
    input_template: str, conditional on processing="generative"
        Template to use for the generative representation of input.
    output_template: str, conditional on processing="generative"
        Template to use for the generative representation of output.
    tokenizer: str
        Name of the tokenizer used to process text data.

    """

    def __init__(self: Dataset, config: t5.config.Config) -> None:
        """Initialize a socio-political event dataset."""
        self.config: t5.config.Config = config
        self.config.declare_option("name", str, "Name of the dataset to train on. It must be a directory in $DATA_PATH.")
        self.config.declare_option("processing", str, "How to process the data, must be one of \"none\", \"generative\".", default="none")
        if self.config.processing == "generative":
            self.config.declare_option("input_template", str, "Template to use for the generative representation of input.")
            self.config.declare_option("output_template", str, "Template to use for the generative representation of output.")
        self.config.declare_option("tokenizer", str, "Name of the tokenizer used to process text data.")
        self.tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(self.config.tokenizer)
        self._initialize_original_splits()
        self._initialize_processed_splits()
        self._initialize_recovery()

    @property
    def train(self: Dataset) -> datasets.Dataset:
        """Processed train split."""
        return self.datasets["train"]

    @property
    def validation(self: Dataset) -> datasets.Dataset:
        """Processed validation split."""
        return self.datasets["validation"]

    @property
    def test(self: Dataset) -> datasets.Dataset:
        """Processed test split."""
        return self.datasets["test"]

    def _initialize_original_splits(self: Dataset) -> None:
        self.original: dict[str, datasets.Dataset] = {}
        for directory in (self.config.DATA_PATH / self.config.name).iterdir():
            self.original[directory.name] = datasets.load_from_disk(str(directory))

    def _process_sample_generative(self: Dataset, sample: dict[str, Any]) -> dict[str, Any]:
        """Process sample as it comes from UCDP into text templates, and tokenize them for training dataset."""
        # init huggingface container for sample
        new_fields: dict[str, Any] = {}

        # parse input & output into plaintext templates from json ucdp sample
        new_fields["input_text"] = self.config.input_template.format(**sample)
        new_fields["output_text"] = self.config.output_template.format(**sample, **new_fields)

        # tokenize inputs+outputs & add result to dict
        new_fields.update({key: value[0] for key, value in self.tokenizer(text=new_fields["input_text"], text_target=new_fields["output_text"], padding="longest", truncation=True, return_tensors="pt").items()})
        return new_fields

    def _make_split_batchable(self: Dataset, split: datasets.Dataset) -> datasets.Dataset:
        return split

    def _process_split(self: Dataset, split: datasets.Dataset) -> datasets.Dataset:
        if self.config.processing == "none":
            return split
        if self.config.processing == "generative":
            return split.map(self._process_sample_generative)  # , load_from_cache_file=False)
        raise NotImplementedError(f"Unknown value for {self.config.key_path('processing')}: {self.config.processing}")

    def _initialize_processed_splits(self: Dataset) -> None:
        self.datasets: dict[str, datasets.Dataset] = {}
        split: str
        for split in ["train", "validation", "test"]:
            if split in self.original:
                self.datasets[split] = self._make_split_batchable(self._process_split(self.original[split]))

    def _initialize_recovery(self: Dataset) -> None:
        if self.config.processing == "generative":
            pattern: str = re.sub(r"\s*{[^}]+}\s*", "(.*?)", self.config.output_template)
            self.template_regex: re.Pattern[str] = re.compile(pattern)
            self.template_fields: list[str] = re.findall(r"{([^}]+)}", self.config.output_template)

    def recover_data_from_output(self: Dataset, sample: dict[str, Any], output: Any) -> dict[str, Any]:
        """Given the output of a model, recover the data from which it was generated."""
        if self.config.processing == "generative":
            indices: list[int] = output.tolist()
            # T5 is using pad as an initial token when generating
            if indices and indices[0] == self.tokenizer.pad_token_id:
                indices = indices[1:]
            with contextlib.suppress(ValueError):
                # ValueError = no EOS token, this will certainly not match the template regex, try it anyway in case we generated exactly the maximum length.
                indices = indices[:indices.index(self.tokenizer.eos_token_id)]
            values: re.Match[str] | None = self.template_regex.fullmatch(self.tokenizer.decode(indices))
            if not values:
                return {}
            # Convert decoded and matched output template fields to dictionary
            parsed_values: dict[str, Any] = dict(zip(self.template_fields, (value.strip() for value in values.groups())))
            return parsed_values
        return {}
