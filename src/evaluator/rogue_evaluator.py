"""
Evaluator for Rogue score.

See https://github.com/pltrdy/rouge
"""
from rouge import Rouge

from logger.logger import Logger
from utils.tokenizer import tokenize


def eval_rogue_score(qa_pairs: list[tuple[list[str], str]]) -> list[tuple[float, float, float]]:
    """
    Evaluates the ROUGE score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

    Returns:
        rouge_score (float): A tuple with the averaged ROUGE score, precision, and recall for all pairs
    """

    def compute_avg_scores(scores):
        f1_scores = [score[0] for score in scores]
        precision_scores = [score[1] for score in scores]
        recall_scores = [score[2] for score in scores]

        avg_f1 = sum(f1_scores) / len(f1_scores)
        avg_precision = sum(precision_scores) / len(precision_scores)
        avg_recall = sum(recall_scores) / len(recall_scores)

        return avg_f1, avg_precision, avg_recall

    rouge_1_scores = compute_avg_scores([rouge_score(gt, a)[0] for (gt, a) in qa_pairs])
    rouge_2_scores = compute_avg_scores([rouge_score(gt, a)[1] for (gt, a) in qa_pairs])

    return [
        rouge_1_scores,
        rouge_2_scores
    ]


def rouge_score(expected: list[str], actual: str) -> list[tuple[float, float, float]]:
    """
    Evaluates the ROUGE score between the ground truth answer and the model's answer.

    Args:
        expected (list[str]): the ground truth possible answers
        actual (str): the model's answer

    Returns:
        rouge_score (list): A list of tuple with the ROUGE score, precision, and recall for the pair
    """
    rouge = Rouge(exclusive=False, metrics=['rouge-1', 'rouge-2'])

    def compute_score(e: str, a: str) -> list[tuple[float, float, float]]:
        expected_tokens = tokenize(e)
        actual_tokens = tokenize(a)

        if len(actual_tokens) == 0 or len(expected_tokens) == 0:
            return [(0, 0, 0), (0, 0, 0)]

        score = rouge.get_scores(
            ' '.join(actual_tokens), ' '.join(expected_tokens))

        return [
            (score[0]['rouge-1']['f'], score[0]
             ['rouge-1']['p'], score[0]['rouge-1']['r']),
            (score[0]['rouge-2']['f'], score[0]
             ['rouge-2']['p'], score[0]['rouge-2']['r']),
        ]

    scores = max((compute_score(e, actual)
                  for e in expected), key=lambda x: x[0][0])

    Logger().debug(
        f"ROUGE score: {scores[0][0]} - Expected: {expected} - Actual: {actual}")
    Logger().debug(
        f"ROUGE score (2): {scores[1][0]} - Expected: {expected} - Actual: {actual}")

    return scores
