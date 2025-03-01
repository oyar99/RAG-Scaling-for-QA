
"""Evaluator for exact match score."""


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

def exact_match(ground_truth: str, answer: str) -> float:
    """
    Evaluates the exact match between the ground truth answer and the model's answer.

    Args:
        ground_truth (str): the ground truth answer
        answer (str): the model's answer

    Returns:
        float: the exact match score
    """
    return 1.0 if ground_truth.lower() == answer.lower() else 0.0
    