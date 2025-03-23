"""Module to queue a batch job for Question Answering using Azure OpenAI."""

import json
import io
import os
import time
from typing import Optional
from openai.types import Batch
from logger.logger import Logger
from azure_open_ai.openai_client import OpenAIClient
from models.agent import Agent
from models.dataset import Dataset

from utils.byte_utils import format_size
from utils.token_utils import estimate_cost, estimate_num_tokens


def guard_job(cost: int) -> None:
    """
    Guard the job based on the estimated cost.

    Args:
        cost (int): the estimated cost of the job
        stop (bool, optional): whether to stop the job if the cost exceeds a certain threshold. Defaults to False.

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

    if cost > 0.25:
        Logger().error(
            "Cost likely exceeds $0.25. Stopping execution ..."
        )
        raise RuntimeError("Program terminated forcefully.")


# pylint: disable-next=too-many-locals
def queue_qa_batch_job(
    model: str,
    dataset: Dataset,
    agent: Agent,
    job_args: dict = None,
    stop: bool = False
) -> Optional[Batch]:
    """
    Queues a batch job for Question Answering using Azure OpenAI.

    Args:
        model (str): the deployment model name
        dataset (Dataset): the dataset to use for the job
        agent (Agent): the agent to use for the job
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
    if not isinstance(dataset, Dataset):
        raise ValueError("dataset must be an instance of Dataset.")

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

    questions = dataset.get_questions()

    prompts = {}

    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'retrieval_jobs')
    output_name = os.path.join(
        output_dir, f'retrieval_results_{Logger().get_run_id()}.jsonl')

    results = []

    all_questions = [q for _, question_set in questions.items()
                     for q in question_set]

    notebooks = agent.multiprocessing_reason(
        questions=[q['question'] for q in all_questions])

    results = [({'custom_id': question["question_id"],
                 'question': question['question'],
                 'result': result.get_sources()}, result.get_notes() + question["question"])
               for result, question in zip(notebooks, all_questions)]

    with open(output_name, 'w', encoding='utf-8') as f:
        for result_json, _ in results:
            r = json.dumps(result_json)
            f.write(r + '\n')

    if stop:
        Logger().warn("Returning without queuing job.")
        return None

    for result_json, prompt in results:
        prompts[result_json['custom_id']] = prompt
        token_count = estimate_num_tokens(prompt, model)
        cost += estimate_cost(token_count, model)
        if token_count > 15000:
            Logger().warn(
                "Prompt for question exceeds 15,000 tokens. Truncation is recommended.")

    guard_job(cost)

    if not isinstance(model, str) or len(model) <= 0:
        raise ValueError(
            "model must be a non-empty string.")

    jobs = [
        {
            "custom_id": question["question_id"],
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": prompts[question["question_id"]]},
                    {"role": "user", "content": question["question"]}
                ],
                "temperature": job_args['temperature'],
                "frequency_penalty": job_args['frequency_penalty'],
                "presence_penalty": job_args['presence_penalty'],
                "max_tokens": job_args['max_tokens']
            },
        }
        for _, question_set in questions.items()
        for question in question_set
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
