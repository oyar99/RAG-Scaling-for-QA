"""Bert Evaluator module."""

from bert_score import score

from logger.logger import Logger
from utils.tokenizer import normalize


def eval_bert_score(qa_pairs: list[tuple[list[str], str]]) -> float:
    """
    Evaluates the BERT score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

    Returns:
        bert_score (float): the BERT score
    """
    return sum(bert_score(gt, a) for (gt, a) in qa_pairs) / len(qa_pairs)


def bert_score(expected: list[str], actual: str) -> float:
    """
    Evaluates the BERT score between the ground truth answer and the model's answer.

    Args:
        expected (list[str]): the ground truth answer
        actual (str): the model's answer

    Returns:
        bert_score (float): the BERT score
    """
    def compute_score(e: str, a: str):
        expected_tokens = normalize(e)
        actual_tokens = normalize(a)

        # pylint: disable-next=unbalanced-tuple-unpacking
        (_, _, f1) = score([actual_tokens], [expected_tokens],
                        lang='en', verbose=False)

        return f1.mean()

    s = max(compute_score(e, actual) for e in expected)

    Logger().debug(
        f"BERT score: {s} - Expected: {expected} - Actual: {actual}")

    return s
