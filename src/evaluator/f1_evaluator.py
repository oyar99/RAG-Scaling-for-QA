"""
Evaluator for F1 score.

Adapted from the MRQA-Shared-Task-2019

See https://mrqa.github.io/
"""

from collections import Counter
from evaluator.normalizer import normalize_answer
from logger.logger import Logger


def eval_f1_score(qa_pairs: list[tuple[str, str]]) -> float:
    """
    Evaluates the F1 score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[str, str]]): the ground truth answers and the model's answers

    Returns:
        float: the F1 score
    """
    return sum(f1_score(gt, a) for (gt, a) in qa_pairs) / len(qa_pairs)


def f1_score(expected: str, actual: str) -> float:
    """
    Evaluates the F1 score between the ground truth answer and the model's answer.

    Args:
        expected (str): the ground truth answer
        actual (str): the model's answer

    Returns:
        float: the F1 score
    """
    expected_tokens = normalize_answer(expected).split()
    actual_tokens = normalize_answer(actual).split()

    common = Counter(expected_tokens) & Counter(actual_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(actual_tokens)
    recall = num_same / len(expected_tokens)

    f1 = (2 * precision * recall) / (precision + recall)

    Logger().debug(
        f"F1 score: {f1} - Expected: {expected} - Actual: {actual}")

    return f1
