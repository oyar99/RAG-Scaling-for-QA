"""Module to queue a batch job for Question Answering using Azure OpenAI."""

import json
import io
import time
from logger.logger import Logger
from azure_open_ai.openai_client import OpenAIClient
from models.question import Question

from utils.byte_utils import format_size
from utils.token_utils import estimate_num_tokens


def queue_qa_batch_job(
    model: str,
    system_prompt: dict[str, str],
    questions: list[Question],
    temperature: float = 0.0,
    max_tokens: int = 1000,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
):
    """
    Queues a batch job for Question Answering using Azure OpenAI.

    Args:
        model (str): the deployment model name
        system_prompt (dict[str, str]): a dictionary that maps conversations to their corresponding system prompts
        questions (list[Question]): the list of questions to be answered
        temperature (float, optional): the sampling temperature to use. Defaults to 0.0.
        max_tokens (int, optional): the maximum number of tokens that can be generated in the completion. 
            Defaults to 1000.
        frequency_penalty (float, optional): penalize new tokens based on their existing frequency in the text so far. 
            Defaults to 0.0.
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

    if not isinstance(temperature, float) or temperature < 0 or temperature > 2:
        raise ValueError("temperature must be a float between 0 and 2")

    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")

    if not isinstance(frequency_penalty, float) or frequency_penalty < -2 or frequency_penalty > 2:
        raise ValueError("frequency_penalty must be a float between -2 and 2")

    if not isinstance(presence_penalty, float) or presence_penalty < -2 or presence_penalty > 2:
        raise ValueError("presence_penalty must be a float between -2 and 2")

    for question in questions:
        token_count = estimate_num_tokens(
            system_prompt[question["conversation_id"]] + question["question"], model)

        if token_count > 20e3:
            Logger().warn(
                f"Question {question['question_id']} exceeds 20,000 tokens.")

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
                "temperature": temperature,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens
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
        if file.status == "processed" or file.status == "error":
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
