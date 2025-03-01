
"""Evaluator for exact match score."""


from evaluator.normalizer import normalize_answer
from logger.logger import Logger


def eval_exact_match(qa_pairs: list[tuple[str, str]]) -> float:
    """
    Evaluates the exact match between the ground truth answers and the model's answers.

    Args:
        ground_truth (list[str]): the ground truth answers
        answers (list[str]): the model's answers

    Returns:
        float: the exact match score
    """
    return sum(exact_match(gt, a) for (gt, a) in qa_pairs) / len(qa_pairs)


def exact_match(expected: str, actual: str) -> float:
    """
    Evaluates the exact match between the ground truth answer and the model's answer.

    Args:
        expected (str): the ground truth answer
        actual (str): the model's answer

    Returns:
        float: the exact match score
    """
    em = 1.0 if normalize_answer(expected) == normalize_answer(actual) else 0.0

    Logger().debug(
        f"Exact match score: {em} - Expected: {expected} - Actual: {actual}")

    return em
