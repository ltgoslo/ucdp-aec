"""
Module for handling socio-political event extraction evaluation.

All metrics are uniquely identified by a path with the following components:
    split_dataset/field/metric
"""

from __future__ import annotations

import abc
import contextlib
import datetime
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.utils.tensorboard.writer import SummaryWriter


if TYPE_CHECKING:
    import pathlib

    import transformers

    import t5.config
    import t5.dataset
    import t5.model


class Metrics:
    """
    A collection of metrics to evaluate socio-political event extraction models.

    This class act as a kind of Metric factory, where instantiating a Metrics object instantiate all registered Metric classes.
    """

    metric_classes: ClassVar[dict[str, type[Metric]]] = {}

    def __init__(self: Metrics, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model, modeldir: pathlib.Path) -> None:
        """Initialize socio-political event extraction metrics."""
        self.config: t5.config.Config = config
        self.dataset: t5.dataset.Dataset = dataset
        self.model: t5.model.Model = model
        self.modeldir: pathlib.Path = modeldir  # for tensorboard logs
        self.metrics: list[Metric] = []
        for metric in Metrics.metric_classes.values():
            self.metrics.append(metric(self.config, self.dataset, self.model))

    @classmethod
    def register(cls: type[Metrics], metric: type[Metric]) -> type[Metric]:
        """Register a Metric class."""
        cls.metric_classes[metric.label()] = metric
        return metric

    def _zero_metrics(self: Metrics) -> None:
        for metric in self.metrics:
            metric.zero()

    def _update_metrics(self: Metrics, evaluation_output: transformers.EvalPrediction) -> None:
        batch_size: int = len(evaluation_output.inputs["id"])
        predictions: torch.Tensor = self.model.detect_prediction_generation(evaluation_output)
        for batch_id in range(batch_size):
            # evaluation_output.label_ids can be accessed through inputs["labels"]
            inputs: dict[str, Any] = { key: value[batch_id] for key, value in evaluation_output.inputs.items() }
            generated_text: torch.Tensor = predictions[batch_id]
            outputs: dict[str, Any] = self.dataset.recover_data_from_output(inputs, generated_text)
            for metric in self.metrics:
                metric(inputs, outputs)

    def _compute_metrics(self: Metrics) -> dict[str, float]:
        values: dict[str, float] = {}
        aggregate_accuracy: list[float] = []
        for metric in self.metrics:
            for key, value in metric.compute().items():
                values[f"{self.dataset.config.name}/{key}"] = value
                if key.endswith("/string accuracy"):
                    aggregate_accuracy.append(value)
        values[f"{self.dataset.config.name}/aggregate/mean string accuracy"] = sum(aggregate_accuracy) / len(aggregate_accuracy)
        return values

    def log_graphs(self: Metrics, logdir: pathlib.Path) -> None:
        """Log TensorBoard graphs for all metrics that produce them. Currently only histograms are available."""
        for metric in self.metrics:
            if hasattr(metric, "log_histogram") and metric.log_histogram:  # log all histograms
                metric.log_histogram(logdir)

    @torch.no_grad()
    def __call__(self: Metrics, evaluation_output: transformers.EvalPrediction, *, compute_result: bool) -> None | dict[str, float]:
        """Update and compute metrics for a given prediction."""
        self._update_metrics(evaluation_output)
        if compute_result:
            result: dict[str, float] = self._compute_metrics()
            self.log_graphs(self.modeldir / "runs")
            self._zero_metrics()
            return result
        return None


class Metric(abc.ABC):
    """A single metric to evaluate (typically one) socio-political event field."""

    def __init__(self: Metric, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
        """Initialize the metric."""
        self.config: t5.config.Config = config
        self.dataset: t5.dataset.Dataset = dataset
        self.model: t5.model.Model = model
        self.zero()

    @abc.abstractmethod
    def zero(self: Metric) -> None:
        """Reinitialize the metric."""

    @classmethod
    @abc.abstractmethod
    def label(cls: type[Metric]) -> str:
        """Get the name of the metric."""

    @abc.abstractmethod
    def __call__(self: Metric, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Update the metric."""

    @abc.abstractmethod
    def compute(self: Metric) -> dict[str, float]:
        """Compute the metric."""


for field in ["side_a_name", "side_b_name", "start_date", "end_date", "location_root_name", "location_adm1_name", "location_adm2_name", "location_where_name", "deaths_side_a", "deaths_side_b", "deaths_civilian", "deaths_unknown", "deaths_low", "deaths_high"]:
    @Metrics.register
    class StringAccuracy(Metric):
        """Compute the accuracy."""

        field: str = field

        def __init__(self: StringAccuracy, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
            """Initialize the accuracy counts."""
            super().__init__(config, dataset, model)

        def zero(self: StringAccuracy) -> None:
            """Reinitialize the metric."""
            self.correct: int = 0
            self.total: int = 0

        def __call__(self: StringAccuracy, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
            """Update the metric."""
            self.total += 1
            self.correct += (str(inputs[self.field]) == outputs.get(self.field, None))

        def compute(self: StringAccuracy) -> dict[str, float]:
            """Compute the metric."""
            accuracy: float = self.correct / self.total if self.total != 0 else 0.
            return { self.label(): accuracy }

        @classmethod
        def label(cls: type[StringAccuracy]) -> str:
            """Get the name of the metric."""
            return f"{cls.field}/string accuracy"


for field in ["deaths_side_a", "deaths_side_b", "deaths_civilian", "deaths_unknown", "deaths_low", "deaths_high"]:
    @Metrics.register
    class RMSE(Metric):
        """Compute the RMSE."""

        field: str = field

        def __init__(self: RMSE, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
            """Initialize the error counts."""
            super().__init__(config, dataset, model)

        def zero(self: RMSE) -> None:
            """Reinitialize the metric."""
            self.squared_error: int = 0
            self.total: int = 0

        def __call__(self: RMSE, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
            """Update the metric."""
            self.total += 1
            value: int = 0
            with contextlib.suppress(ValueError): # if it's not a number assume 0
                value = int(outputs.get(self.field, "0"))
            self.squared_error += (inputs[self.field] - value)**2

        def compute(self: RMSE) -> dict[str, float]:
            """Compute the metric."""
            rmse: float = (self.squared_error / self.total)**0.5
            return { self.label(): rmse }

        @classmethod
        def label(cls: type[RMSE]) -> str:
            """Get the name of the metric."""
            return f"{cls.field}/RMSE"


for field in ["start_date", "end_date"]:
    @Metrics.register
    class DateRMSE(Metric):
        """Compute the RMSE (in days) for YYYY-MM-DD dates."""

        field: str = field

        def __init__(self: DateRMSE, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
            """Initialize the error counts."""
            super().__init__(config, dataset, model)

        def zero(self: DateRMSE) -> None:
            """Reinitialize the metric."""
            self.squared_error: int = 0
            self.total: int = 0
            self.unparsable: int = 0
            self.day_differences: list[int] = []

        def __call__(self: DateRMSE, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
            """Update the metric."""
            # add to sample count
            self.total += 1

            # declare input publication date (input_source_date) for default values
            input_source_date: Any = inputs.get("source_date")
            input_start_date: Any = inputs.get(self.field)
            if not isinstance(input_source_date, datetime.datetime) or not isinstance(input_start_date, datetime.date):  # if faulty input, ignore
                return
            default_date: datetime.date = input_source_date.date()  # remove hh:mm:ss info
            output_start_date: Any = outputs.get(self.field)

            # init dates
            sample: datetime.date = input_start_date
            value: datetime.date = default_date

            # attempt parsing predicted start_date, if not possible, it will be the sample source_date
            try:
                value = datetime.datetime.strptime(output_start_date, "%Y-%m-%d").date()
            except (ValueError, TypeError):
                self.unparsable += 1

            # calculate and store differences
            day_difference = (sample - value).days
            self.day_differences.append(day_difference)
            self.squared_error += day_difference ** 2

        def compute(self: DateRMSE) -> dict[str, float]:
            """Compute the metric."""
            rmse: float = (self.squared_error / self.total)**0.5
            return { self.label(): rmse, "debug/start_date/unparsable": self.unparsable / self.total}

        def log_histogram(self: DateRMSE, logdir: pathlib.Path) -> None:
            """Log the histogram of differences using TensorBoard."""
            writer: Any = SummaryWriter(logdir)
            writer.add_histogram(
                f"error_metrics/{self.dataset.config.name}/date_differences",
                torch.tensor(self.day_differences)
            )
            writer.close()

        @classmethod
        def label(cls: type[DateRMSE]) -> str:
            """Get the name of the metric."""
            return f"{cls.field}/date RMSE in days"


@Metrics.register
class ParsableOutput(Metric):
    """Verify whether the output of the model respect the template."""

    def __init__(self: ParsableOutput, config: t5.config.Config, dataset: t5.dataset.Dataset, model: t5.model.Model) -> None:
        """Initialize the accuracy counts."""
        super().__init__(config, dataset, model)

    def zero(self: ParsableOutput) -> None:
        """Reinitialize the metric."""
        self.correct: int = 0
        self.total: int = 0

    def __call__(self: ParsableOutput, _: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Update the metric."""
        self.total += 1
        self.correct += (len(outputs) > 1) # contains at least the "logits" field

    def compute(self: ParsableOutput) -> dict[str, float]:
        """Compute the metric."""
        accuracy: float = self.correct / self.total if self.total != 0 else 0.
        return { self.label(): accuracy }

    @classmethod
    def label(cls: type[ParsableOutput]) -> str:
        """Get the name of the metric."""
        return "debug/parsable output"
