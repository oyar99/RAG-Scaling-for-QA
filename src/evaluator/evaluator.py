"""Evaluator module."""

import os
from evaluator.exact_match_evaluator import eval_exact_match
from logger.logger import Logger


def evaluator(qa_pairs: list[tuple[str, str]]) -> None:
    """
    Evaluates the exact match between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[str, str]]): the ground truth answers and the model's answers
        metric (str): the metric to be used for evaluation

    Returns:
        float: the evaluation score
    """

    Logger().info("Evaluating exact match score")

    em = eval_exact_match(qa_pairs)

    Logger().info(f"Exact match score: {em}")

    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output')
    output_name = os.path.join(output_dir, f'eval-{Logger().get_run_id()}.out')

    with open(output_name, "w", encoding="utf-8") as output_file:
        output_file.write(f"Exact match score: {em}")
