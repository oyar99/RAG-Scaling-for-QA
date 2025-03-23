"""Judge Evaluator Module"""
import json
from azure_open_ai.batch_evaluation import queue_evaluation_batch_job
from src.logger.logger import Logger


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


def eval_judge_score_with_file(file_path: str) -> float:
    """
    Evaluate the question answer pairs based on a score given by an LLM judge.
    Reads a batch results file from Azure OpenAI for evaluation.

    Args:
        file_path (str): The path to the file where the evaluation results were saved.

    Returns:
        score (float): The computed score based on the evaluation results.
    """
    total = 0
    yes = 0
    no = 0

    with open(file_path, 'r', encoding='utf-8') as evaluation_file:
        evaluation = [json.loads(line) for line in evaluation_file]

        for eval_item in evaluation:
            if 'content' not in eval_item['response']['body']['choices'][0]['message']:
                Logger().warn(
                    "Content not found in the response. Skipping evaluation ...")
                continue

            total += 1
            content = str(eval_item['response']['body']
                          ['choices'][0]['message']['content'])

            if content.lower().strip() == 'yes':
                yes += 1
            elif content.lower().strip() == 'no':
                no += 1
            else:
                Logger().warn(
                    f"Unexpected content in the response: {content}. Treating as 'no'.")
                continue

    if total > 0:
        score = yes / total
        Logger().info(
            f"Evaluation completed. Total: {total}, Yes: {yes}, No: {no}, Score: {score:.2f}")
        return score

    return 0.0
