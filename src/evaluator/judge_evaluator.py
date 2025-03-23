"""Judge Evaluator Module"""
from azure_open_ai.batch_evaluation import queue_evaluation_batch_job


def eval_judge_score(qa_pairs: list[tuple[str, str, str]]) -> None:
    """
    Evaluate the question answer pairs based on a score given by an LLM judge.
    Uploads a batch job to Azure OpenAI for evaluation.
    The evaluation is done by comparing the expected answer with the provided answer.

    Args:
        qa_pairs (list[tuple[str, str]]): A list of tuples containing question, expected answer and given answer pairs.
        file_path (str): The path to the file where the evaluation results were saved.
    """
    queue_evaluation_batch_job(
        model='gpt-4o-mini',
        question_answers=qa_pairs,
    )


def eval_judge_score_with_file(file_path: str) -> None:
    """
    Evaluate the question answer pairs based on a score given by an LLM judge.
    Reads a batch results file from Azure OpenAI for evaluation.

    Args:
        file_path (str): The path to the file where the evaluation results were saved.
    """
