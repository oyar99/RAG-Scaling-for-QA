"""
Evaluator for BLEU score
"""
import evaluate


def eval_bleu_score(qa_pairs: list[tuple[list[str], str]]) -> float:
    """
    Evaluate the BLEU score of the given question-answer pairs.

    Args:
        qa_pairs (list[tuple[list[str], str]]): A list of tuples where each tuple contains a list of 
        candidate answers and a reference answer.

    Returns:
        float: the BLEU score.
    """
    bleu = evaluate.load("bleu")
    results = []
    for references, candidate in qa_pairs:
        result = max(bleu.compute(predictions=[candidate], references=[ref])['bleu'] for ref in references)
        results.append(result)

    avg_bleu = sum(results) / len(results)
    return avg_bleu
