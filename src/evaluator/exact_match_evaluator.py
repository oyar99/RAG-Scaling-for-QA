
"""Evaluator for exact match score."""


from utils.tokenizer import normalize
from logger.logger import Logger


def eval_exact_match(qa_pairs: list[tuple[list[str], str]]) -> float:
    """
    Evaluates the exact match between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

    Returns:
        em (float): the exact match score
    """
    return sum(exact_match(gt, a) for (gt, a) in qa_pairs) / len(qa_pairs)


def exact_match(expected: list[str], actual: str) -> float:
    """
    Evaluates the exact match between the ground truth answer and the model's answer.

    Args:
        expected (list[str]): the ground truth answer
        actual (str): the model's answer

    Returns:
        em (float): the exact match score
    """
    em = max(
        1.0 if normalize(expected_instance) == normalize(actual)
        else 0.0
        for expected_instance in expected
    )

    Logger().debug(
        f"Exact match score: {em} - Expected: {expected} - Actual: {actual}")

    return em
