"""Metrics evaluator module"""

from logger.logger import Logger


def eval_metrics(metrics: list[dict[str, int]]) -> dict[str, float]:
    """
    Evaluates the metrics across all samples.

    Args:
        metrics (list[dict[str, int]]): A list of dictionaries containing the metrics for each sample.

    Returns:
        dict[str, float]: A dictionary containing the average of each metric.
    """
    if not metrics:
        Logger().error("Metrics list is empty.")
        raise ValueError("Metrics list is empty.")

    metric_keys = [
        'completion_tokens',
        'prompt_tokens',
        'total_tokens',
    ]

    # Compute the total for each metric
    total_metrics = {metric: 0 for metric in metric_keys}
    for sample in metrics:
        for metric in metric_keys:
            total_metrics[metric] += sample.get(metric, 0)

    Logger().info(f"Total metrics: {total_metrics}")

    avg_metrics = {}
    for metric in metric_keys:
        avg_metrics[metric] = sum(
            sample[metric] for sample in metrics) / len(metrics)

    Logger().info(f"Average metrics: {avg_metrics}")

    return avg_metrics
