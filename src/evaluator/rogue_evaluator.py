"""
Evaluator for Rogue score.

See https://github.com/pltrdy/rouge
"""
from rouge import Rouge

from utils.tokenizer import tokenize


def eval_rogue_score(qa_pairs: list[tuple[list[str], str]]) -> list[tuple[float, float, float]]:
    """
    Evaluates the ROUGE score between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

    Returns:
        rouge_score (float): A tuple with the averaged ROUGE score, precision, and recall for all pairs
    """

    rouge = Rouge()

    try:
        scores = [max(rouge.get_scores(' '.join(tokenize(a)), ' '.join(tokenize(ref))),
                      key=lambda score: score['rouge-1']['f'])
                  for (gt, a) in qa_pairs for ref in gt if len(tokenize(a)) > 0 and len(tokenize(ref)) > 0]
    # pylint: disable-next=broad-exception-caught
    except Exception:
        scores = [{'rouge-1': {'f': 0.0, 'r': 0.0, 'p': 0.0},
                   'rouge-2': {'f': 0.0, 'r': 0.0, 'p': 0.0}, 'rouge-l': {'f': 0.0, 'r': 0.0, 'p': 0.0}}]

    # Calculate average scores for rouge-1
    avg_rouge = sum(score['rouge-1']['f'] for score in scores) / len(scores)
    avg_precision = sum(score['rouge-1']['p']
                        for score in scores) / len(scores)
    avg_recall = sum(score['rouge-1']['r'] for score in scores) / len(scores)

    # Calculate average scores for rouge-2
    avg_rouge_2 = sum(score['rouge-2']['f'] for score in scores) / len(scores)
    avg_precision_2 = sum(score['rouge-2']['p']
                          for score in scores) / len(scores)
    avg_recall_2 = sum(score['rouge-2']['r'] for score in scores) / len(scores)
    # Calculate average scores for rouge-l
    avg_rouge_l = sum(score['rouge-l']['f'] for score in scores) / len(scores)
    avg_precision_l = sum(score['rouge-l']['p']
                          for score in scores) / len(scores)
    avg_recall_l = sum(score['rouge-l']['r'] for score in scores) / len(scores)

    return [(avg_rouge, avg_precision, avg_recall),
            (avg_rouge_2, avg_precision_2, avg_recall_2),
            (avg_rouge_l, avg_precision_l, avg_recall_l)]
