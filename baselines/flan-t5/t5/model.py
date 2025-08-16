"""Module for handling socio-political models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import transformers
from transformers import EvalPrediction

import t5.config
import t5.dataset


class ModelConfig(transformers.PretrainedConfig):
    """Binding class between huggingface config and our own."""

    def __init__(self: ModelConfig, config: t5.config.Config | dict[str, dict[str, str]] | None = None, **kwargs: Any) -> None:
        """Initialize a ModelConfig either from a config or from a loaded json object."""
        if isinstance(config, t5.config.Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = t5.config.Config(root=["model"], lazy_values = config["config"])
        else:
            self.config = t5.config.Config(root=["model"])
        super().__init__(**kwargs)

    def to_dict(self: ModelConfig) -> dict[str, Any]:
        """Convert the configuration to a json object using strings for all values."""
        return {"config": {
                key: str(value)
                for key, value in self.config.items()
            }}


class Model(transformers.PreTrainedModel):
    """
    A socio-political event extraction model.

    Config
    ------
    transformer: str
        Name of the huggingface model to use.

    generation_max_length: int
        Maximum number of tokens to generate during evaluetion.

    """

    config_class = ModelConfig

    def __init__(self: Model, config: t5.config.Config, _: t5.dataset.Dataset) -> None:
        """Initialize a socio-political event extraction model."""
        self.config = ModelConfig(config)
        super().__init__(self.config)
        config.declare_option("transformer", str, "Name of the huggingface model to use.")
        self.seq2seq = transformers.AutoModelForSeq2SeqLM.from_pretrained(config.transformer)
        config.declare_option("generation_max_length", int, "Maximum number of tokens to generate during evaluetion.")
        self.seq2seq.generation_config.max_length = config.generation_max_length
        self._keys_to_ignore_on_save = None

    def forward(self: Model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> transformers.modeling_outputs.Seq2SeqLMOutput:
        """Return the loss for a given input and gold output."""
        del kwargs
        output: transformers.modeling_outputs.Seq2SeqLMOutput = self.seq2seq(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def generate(self: Model, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Extract the information for a given input."""
        del kwargs
        output: torch.Tensor = self.seq2seq.generate(input_ids=input_ids, attention_mask=attention_mask)
        return output

    @staticmethod
    def detect_prediction_generation(output: EvalPrediction) -> Any:
        """Detect if the model output was predicted with generate() or not."""
        predictions = output.predictions
        if isinstance(predictions, torch.Tensor):
            return predictions
        if isinstance(predictions, tuple):
            return torch.argmax(predictions[0], dim=-1)