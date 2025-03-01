"""Bert Evaluator module."""

from bert_score import score

from evaluator.normalizer import normalize_answer
from logger.logger import Logger


def eval_bert_score(qa_pairs: list[tuple[str, str]]) -> float:
    """
    Evaluates the BERT score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[str, str]]): the ground truth answers and the model's answers

    Returns:
        float: the BERT score
    """
    return sum(bert_score(gt, a) for (gt, a) in qa_pairs) / len(qa_pairs)


def bert_score(expected: str, actual: str) -> float:
    """
    Evaluates the BERT score between the ground truth answer and the model's answer.

    Args:
        expected (str): the ground truth answer
        actual (str): the model's answer

    Returns:
        float: the BERT score
    """
    expected_tokens = normalize_answer(expected)
    actual_tokens = normalize_answer(actual)

    # pylint: disable-next=unbalanced-tuple-unpacking
    (_, _, f1) = score([actual_tokens], [expected_tokens],
                     lang='en', verbose=False)

    s = f1.mean()

    Logger().debug(
        f"BERT score: {s} - Expected: {expected} - Actual: {actual}")

    return s
