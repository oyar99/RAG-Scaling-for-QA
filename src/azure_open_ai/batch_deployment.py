"""Module to queue a batch job for Question Answering using Azure OpenAI."""

import json
import io
import time
from logger.logger import Logger
from azure_open_ai.openai_client import OpenAIClient
from models.question import Question

from utils.byte_utils import format_size
from utils.token_utils import estimate_cost, estimate_num_tokens


def guard_job(cost: int, stop: bool = False) -> None:
    """Guard the job based on the estimated cost.

    Args:
        cost (int): the estimated cost of the job

    Raises:
        RuntimeError: if the cost exceeds $2.0
    """
    if cost == 0.0:
        Logger().error("Estimated cost is $0.0. Please review the questions.")
        raise RuntimeError("Program terminated forcefully.")

    if cost > 0.0:
        Logger().info(f"Estimated cost: {cost:.2f}")

    if cost > 0.1:
        Logger().warn(
            "Estimated cost exceeds $0.1. \
Please review the questions and ensure they are not too verbose.")

    if cost > 0.25 and not stop:
        Logger().error(
            "Cost likely exceeds $0.25. Stopping execution ..."
        )
        raise RuntimeError("Program terminated forcefully.")


# pylint: disable-next=too-many-locals
def queue_qa_batch_job(
    model: str,
    system_prompt: dict[str, str],
    questions: list[Question],
    job_args: dict = None,
    stop: bool = False
):
    """
    Queues a batch job for Question Answering using Azure OpenAI.

    Args:
        model (str): the deployment model name
        system_prompt (dict[str, str]): a dictionary that maps conversations to their corresponding system prompts
        questions (list[Question]): the list of questions to be answered
        job_args (dict, optional): a dictionary of job arguments. Defaults to None. Possible arguments are:
            temperature (float, optional): the sampling temperature to use. Defaults to 0.0.
            max_tokens (int, optional): the maximum number of tokens that can be generated in the completion.
            Defaults to 1000.
            frequency_penalty (float, optional): penalize new tokens based on their existing frequency in the 
            text so far. Defaults to 0.0.
            presence_penalty (float, optional): penalize new tokens based on whether they appear in the text so far. 
            Defaults to 0.0.

    Raises:
        ValueError: if any of the input parameters are invalid
        RuntimeError: if the file upload fails
    """

    if not isinstance(model, str) or len(model) <= 0:
        raise ValueError(
            "model must be a non-empty string.")

    if not isinstance(system_prompt, dict) or len(system_prompt) <= 0:
        raise ValueError("system_prompt must be a non-empty string.")

    if not isinstance(questions, list) or len(questions) <= 0:
        raise ValueError("questions must be a non-empty list.")

    if job_args is None:
        job_args = {
            'temperature': 0.0,
            'max_tokens': 50,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }

    if not isinstance(job_args, dict):
        raise ValueError("job_args must be a dictionary.")

    cost = 0.0

    for question in questions:
        token_count = estimate_num_tokens(
            system_prompt[question["conversation_id"]] + question["question"], model)

        cost += estimate_cost(token_count, model)

        if token_count > 15000:
            Logger().warn(
                f"Question {question['question_id']} exceeds 15,000 tokens. Truncation is recommended.")

    guard_job(cost, stop)

    if stop:
        Logger().warn("Returning without queuing job.")
        return None

    jobs = [
        {
            "custom_id": question["question_id"],
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": system_prompt[question["conversation_id"]]},
                    {"role": "user", "content": question["question"]}
                ],
                "temperature": job_args['temperature'],
                "frequency_penalty": job_args['frequency_penalty'],
                "presence_penalty": job_args['presence_penalty'],
                "max_tokens": job_args['max_tokens']
            },
        }
        for question in questions
    ]

    jobs_jsonl = "\n".join(json.dumps(job) for job in jobs)
    jsonl_encoded = jobs_jsonl.encode("utf-8")

    Logger().info(f"batch file size: {format_size(len(jsonl_encoded))}")

    byte_stream = io.BytesIO(jsonl_encoded)

    Logger().info("Starting batch file upload ...")

    openai_client = OpenAIClient().get_client()

    # Upload jsonl file for batch processing
    batch_file = openai_client.files.create(
        file=(f'locomo-run-{Logger().get_run_id()}.jsonl',
              byte_stream, 'application/jsonl'),
        purpose="batch",
    )

    # Wait until the file is uploaded
    while True:
        file = openai_client.files.retrieve(batch_file.id)
        if file.status in ("processed", "error"):
            break
        Logger().info("Waiting for file to be uploaded...")
        time.sleep(10)

    if file.status == "error":
        # pylint: disable-next=broad-except
        raise RuntimeError(f"File upload failed: {file.error}")

    Logger().info("File upload succeeded.")
    Logger().info("Creating batch job ...")

    # Create a batch job
    batch_job = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    return batch_job
