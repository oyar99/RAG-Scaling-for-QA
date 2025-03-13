"""
Evaluator for F1 score.

Adapted from the MRQA-Shared-Task-2019

See https://mrqa.github.io/
"""

from collections import Counter
from utils.tokenizer import tokenize
from logger.logger import Logger


def eval_f1_score(qa_pairs: list[tuple[list[str], str]]) -> tuple[float, float, float]:
    """
    Evaluates the F1 score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

    Returns:
        f1_score (float): A tuple with the averaged F1 score, precision, and recall for all pairs
    """
    f1_scores = [f1_score(gt, a) for (gt, a) in qa_pairs]
    f1, precision, recall = zip(*f1_scores)

    avg_f1 = sum(f1) / len(f1)
    avg_precision = sum(precision) / len(precision)
    avg_recall = sum(recall) / len(recall)

    return avg_f1, avg_precision, avg_recall


def f1_score(expected: list[str], actual: str) -> tuple[float, float, float]:
    """
    Evaluates the F1 score between the ground truth answer and the model's answer.

    Args:
        expected (str): the ground truth answer
        actual (str): the model's answer

    Returns:
        f1_score (float): A tuple with the F1 score, precision, and recall for the pair
    """
    def compute_score(e: str, a: str) -> tuple[float, float, float]:
        expected_tokens = tokenize(e)
        actual_tokens = tokenize(a)

        common = Counter(expected_tokens) & Counter(actual_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return (0, 0, 0)

        precision = 1.0 * num_same / len(actual_tokens)
        recall = 1.0 * num_same / len(expected_tokens)

        f1 = (2 * precision * recall) / (precision + recall)

        return f1, precision, recall

    f1, precision, recall = max((compute_score(e, actual)
                                for e in expected), key=lambda x: x[0])

    Logger().debug(
        f"F1 score: {f1} - Expected: {expected} - Actual: {actual}")

    return f1, precision, recall
