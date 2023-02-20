from copy import deepcopy
from typing import Any

import torch
import torchmetrics as tm


class Metrics:
    """A wrapper class for logging multiple metrics from
    torchmetrics"""

    def __init__(self, metrics: dict[str, tm.Metric], phases: list[str]) -> None:
        """A class for logging metrics

        :param metrics: A dictionary of {name: Metric class} to log
        :type metrics: dict[str, tm.Metric]
        :param phases: A list of phases to log for, will be shown in wandb this way
        :type phases: list[str]
        """
        self.metrics = metrics
        self.phases = phases

        self.phase_metrics = {phase: {name: deepcopy(metric) for name, metric in self.metrics} for phase in self.phases}

        self.epoch_metrics_container = {phase: {name: 0 for name in self.metrics} for phase in self.phases}
        self.step_metrics_container = {phase: {name: [] for name in self.metrics} for phase in self.phases}

        self.phase_steps = {phase: 0 for phase in self.phases}

    def compute_step(self, phase: str, preds: torch.Tensor, targets: torch.Tensor, log_every_n_steps: int) -> dict[str, Any]:
        assert phase in self.phases, f"Phase {phase} not in phases set at initiliazation"
        curr_step_metrics = None

        # calculate metric no matter what to update internal state
        for metric in self.metrics:
            try:
                r = self.phase_metrics[phase][metric](preds, targets).item()
                self.step_metrics_container[phase][metric] = r
            except AttributeError:
                print(f"Metric {metric} did not return a float, not logging and continuing...")

        # update phase step since we calculated the metric
        self.phase_steps[phase] += 1

        # only return the dict if we're logging on this step
        if self.phase_steps[phase] % log_every_n_steps == 0:
            curr_step_metrics = {f"{phase}_{metric}": self.step_metrics_container[phase][metric] for metric in self.metrics}

        return curr_step_metrics

    def compute_epoch(self, phase: str):
        # store epoch metrics for "best metric" logging
        for metric in self.metrics:
            try:
                r = self.phase_metrics[phase][metric].compute().item()
                self.epoch_metrics_container[phase][metric].append(r)
            except AttributeError:
                print(f"Metric {metric} did not return a float, not logging and continuing...")

        # reset the metric classes for next epoch
        for metric in self.metrics:
            self.epoch_metrics_container[phase][metric].reset()

        curr_epoch_metrics = {f"{phase}_{metric}": self.epoch_metrics_container[phase][metric][-1] for metric in self.metrics}

        return curr_epoch_metrics

    def finish(self, phase):
        best_epoch = {f"{metric}_best_epoch_{phase}": np.argmax()}
