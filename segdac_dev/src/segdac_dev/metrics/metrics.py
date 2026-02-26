import numpy as np


class Metrics:
    def __init__(self):
        self.data = {}

    def accumulate_metric(
        self, metric_name: str, metric_value: float, agg_fn=lambda x: np.array([x])
    ):
        if self.data.get(metric_name) is None:
            self.data[metric_name] = {"values": [metric_value], "agg_fn": agg_fn}
        else:
            self.data[metric_name]["values"].append(metric_value)

    def compute_aggregated_metrics(self) -> dict:
        aggregated_metrics = {}
        for metric_name in self.data.keys():
            values = self.data[metric_name]["values"]
            agg_fn = self.data[metric_name]["agg_fn"]
            aggregated_metrics[metric_name] = agg_fn(values)
        self.data = {}
        return aggregated_metrics
