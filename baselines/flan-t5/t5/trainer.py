"""Define class for model training."""

from __future__ import annotations

import datetime
import itertools
import json
import logging
from typing import TYPE_CHECKING, Any, Iterator

import transformers

import t5.collator
import t5.dataset
import t5.experiment
import t5.metric
import t5.model


if TYPE_CHECKING:
    import pathlib

    import numpy.typing


logger: logging.Logger = logging.getLogger(__name__)


class Trainer(t5.experiment.BaseExperiment):
    """
    Train a model.

    Config
    ------
    dataset: Config
        configuration for the t5.Dataset object

    """

    def init(self: Trainer) -> None:
        """Create objects needed for training."""
        self.config.declare_option("dataset", t5.config.Config, "Configuration of the dataset (see `t5.Dataset`).")
        self.dataset = t5.dataset.Dataset(self.config.dataset)
        self.config.declare_option("transformer", str, "Name of the huggingface model to use.")
        self.config.declare_option("model", t5.config.Config, "Configuration of the model (see `t5.Model`).")
        self.model = t5.model.Model(self.config.model, self.dataset)
        self.data_collator = t5.collator.DataCollator(self.dataset, self.model)
        self.metrics = t5.metric.Metrics(self.config, self.dataset, self.model, self.modeldir)
        self.config.declare_option("num_train_epochs", float, "Number of training epochs.", 18.0)
        self.config.declare_option("learning_rate", float, "The initial learning rate.", 3e-5)
        self.modeldir.mkdir(parents=True, exist_ok=False)
        training_arguments = transformers.Seq2SeqTrainingArguments(
                self.modeldir,
                report_to=["tensorboard"],
                logging_dir=self.modeldir / "runs",
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                bf16=True,
                eval_strategy=transformers.IntervalStrategy.STEPS,
                eval_steps=0.1,
                save_steps=0.1,
                load_best_model_at_end=True,
                save_total_limit=3,
                metric_for_best_model=f"{self.dataset.config.name}/aggregate/mean string accuracy",
                predict_with_generate=False,
                eval_do_concat_batches=False,
                greater_is_better=True,
                num_train_epochs=self.config.num_train_epochs,
                learning_rate=self.config.learning_rate,
                remove_unused_columns=False,
                include_inputs_for_metrics=True,
                batch_eval_metrics=True,
                log_level="info")
        self.trainer = transformers.Seq2SeqTrainer(
                args=training_arguments,
                train_dataset=self.dataset.train,
                eval_dataset=self.dataset.validation,
                model=self.model,
                data_collator=self.data_collator,
                compute_metrics=self.metrics
        )
        self.config.declare_option("eval", bool, "Whether to run only evaluation.", default=False)

    def _write_predictions(self: Trainer, split: str, predictions: Iterator[numpy.typing.NDArray[numpy.int64]]) -> None:
        output_path: pathlib.Path = self.logdir / f"{split}_output.jsonl"
        with output_path.open("w") as file:
            for sample, generated_text in zip(self.dataset.datasets[split], predictions):
                prediction_json: dict[str, Any] = self.dataset.recover_data_from_output(sample, generated_text)
                prediction_json["generated_text"] = self.dataset.tokenizer.decode(generated_text)
                json_sample: dict[str, Any] = {   # write dates in files in string format
                        key: (str(value) if isinstance(value, (datetime.datetime, datetime.date)) else value)
                        for key, value in sample.items()
                        }
                json.dump({"sample": json_sample, "prediction": prediction_json}, file)
                print("\n", end="", file=file)

    def evaluate(self: Trainer, split: str) -> dict[str, float]:
        """Evaluate the model on a given split."""
        self.trainer.args.predict_with_generate = True
        output: transformers.PredictionOutput = self.trainer.predict(self.dataset.datasets[split], metric_key_prefix=split)
        unbatched_predictions: Iterator[numpy.typing.NDArray[numpy.int64]] = itertools.chain.from_iterable(output.predictions)
        self._write_predictions(split, unbatched_predictions)
        metrics: dict[str, float] = output.metrics
        for key, value in metrics.items():
            logger.info("metric\t%s\t%s", key, value)
            print(key, value)
        return metrics

    def run(self: Trainer) -> None:
        """Train."""
        if not self.config.eval:
            self.trainer.train()
        metrics: dict[str, float] = self.evaluate("validation")
        metrics.update(self.evaluate("test"))
        with (self.logdir / "results.json").open("w") as results_file:
            json.dump(metrics, results_file)
