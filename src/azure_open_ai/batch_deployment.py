# pylint: disable=dangerous-default-value

"""Module to queue a batch job for Question Answering using Azure OpenAI."""

import json
import os
from typing import Optional
from openai.types import Batch
from azure_open_ai.batch import queue_batch_job
from logger.logger import Logger
from models.agent import Agent
from models.dataset import Dataset
from utils.model_utils import supports_temperature_param
from utils.token_utils import estimate_cost, estimate_num_tokens, get_max_output_tokens, truncate_content

default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 100,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}


def guard_job(results: list[tuple[dict, str]], model: str, stop: bool) -> None:
    """
    Guard the job based on the estimated cost.

    Args:
        results (list[tuple[dict, str]]): the results of the job
        model (str): the deployment model name
        stop (bool): whether to stop the job

    Raises:
        RuntimeError: if the cost exceeds $2.0
    """
    if not isinstance(model, str) or len(model) <= 0:
        raise ValueError(
            "model must be a non-empty string.")

    if stop:
        Logger().error("Returning without queuing job.")
        raise ValueError(
            "Returning without queuing job. Please check the arguments."
        )

    cost = 0.0

    for _, prompt in results:
        token_count = estimate_num_tokens(prompt, model)
        cost += estimate_cost(token_count, model)
        if token_count > 15000:
            Logger().warn(
                "Prompt for question exceeds 15,000 tokens. Truncation is recommended.")

    if cost == 0.0:
        Logger().error("Estimated cost is $0.0. Please review the questions.")
        raise RuntimeError("Program terminated forcefully.")

    if cost > 0.0:
        Logger().info(f"Estimated cost: {cost:.2f}")

    if cost > 0.4:
        Logger().warn(
            "Estimated cost exceeds $0.4. \
Please review the questions and ensure they are not too verbose.")

    if cost > 10.0:
        Logger().error(
            "Cost likely exceeds $1.0. Stopping execution ..."
        )
        raise RuntimeError("Program terminated forcefully.")


def get_retrieval_output_path() -> str:
    """
    Get the output path for the batch job results.

    Returns:
        str: the output path
    """
    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'retrieval_jobs')
    return os.path.join(
        output_dir, f'retrieval_results_{Logger().get_run_id()}.jsonl')


def get_qa_output_path() -> str:
    """
    Get the output path for the batch job results.

    Returns:
        str: the output path
    """
    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'qa_jobs')
    return os.path.join(
        output_dir, f'qa_results_{Logger().get_run_id()}.jsonl')


def queue_qa_job(
    model: str,
    dataset: Dataset,
    agent: Agent,
    job_args: dict = default_job_args,
    stop: bool = False
) -> Optional[Batch]:
    """
    Queues a job for Question Answering using Azure OpenAI.

    Args:
        model (str): the deployment model name
        dataset (Dataset): the dataset to use for the job
        agent (Agent): the agent to use for the job
        job_args (dict, optional): a dictionary of job arguments. Defaults to None. Possible arguments are:
            temperature (float, optional): the sampling temperature to use. Defaults to 0.0.
            max_tokens (int, optional): this parameter is ignored in the batch job. The maximum number of tokens
            that can be generated in the completion defaults to all possible tokens.
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

    if not isinstance(agent, Agent):
        raise ValueError("agent must be an instance of Agent.")

    if not agent.support_batch:
        Logger().error(
            "agent does not support batch reasoning. Use queue_qa_batch_job instead."
        )
        raise ValueError(
            "agent does not support batch reasoning. Use queue_qa_batch_job instead."
        )

    questions = dataset.get_questions()

    notebook = agent.batch_reason([q['question']
                                   for _, question_set in questions.items()
                                   for q in question_set])

    questions_list = [f'Q ({question["question_id"]}): {question["question"]}'
                      for _, question_set in questions.items()
                      for question in question_set]

    questions_objs = [question for _, question_set in questions.items()
                      for question in question_set]

    batch_size = 5

    question_batches = [
        {
            'content': '\n'.join(questions_list[i:i + batch_size]).strip(),
            'context': truncate_content(
                content=notebook.get_notes(),
                must_have_texts=[doc['content']
                                 for question in questions_objs[i:i + batch_size]
                                 for doc in question['docs']],
                context_starts_idx=notebook.get_actual_context_idx(),
                model=model)
        }
        for i in range(0, len(questions_list), batch_size)
    ]

    results = [({
        'custom_id': f'{Logger().get_run_id()}-{i}',
        'question': question_batch['content'],
        'result': notebook.get_sources()
    }, question_batch['context']) for i, question_batch in enumerate(question_batches)]

    guard_job(results, model, stop)

    return queue_batch_job([
        {
            "custom_id": f'{Logger().get_run_id()}-{i}',
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": question_batch['context']},
                    {"role": "user", "content": question_batch['content']}
                ],
                "temperature": job_args['temperature'] if supports_temperature_param(model) else None,
                "frequency_penalty": job_args['frequency_penalty'],
                "presence_penalty": job_args['presence_penalty'],
                "max_completion_tokens": int(get_max_output_tokens(model)),
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "strict": True,
                        "name": "question_answering",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "result": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question_id": {"type": "string"},
                                            "answer": {"type": "string"},
                                        },
                                        "required": ["question_id", "answer"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["result"],
                            "additionalProperties": False
                        }
                    }
                }
            },
        }
        for i, question_batch in enumerate(question_batches)
    ])

# pylint: disable-next=too-many-locals
def queue_qa_batch_job(
    model: str,
    dataset: Dataset,
    agent: Agent,
    job_args: dict = default_job_args,
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

    if not isinstance(agent, Agent):
        raise ValueError("agent must be an instance of Agent.")

    if agent.support_batch:
        Logger().error(
            "agent does not support batch reasoning. Use queue_qa_job instead."
        )
        raise ValueError(
            "agent does not support batch reasoning. Use queue_qa_job instead."
        )

    questions = dataset.get_questions()

    prompts = {}
    results = []

    all_questions = [q for _, question_set in questions.items()
                     for q in question_set]

    notebooks = agent.multiprocessing_reason(
        questions=[q['question'] for q in all_questions])

    results = [({'custom_id': question["question_id"],
                 'question': question['question'],
                 'result': result.get_sources()}, result.get_notes())
               for result, question in zip(notebooks, all_questions)]

    with open(get_retrieval_output_path(), 'w', encoding='utf-8') as f:
        for result_json, _ in results:
            r = json.dumps(result_json)
            f.write(r + '\n')

    for result_json, prompt in results:
        prompts[result_json['custom_id']] = prompt

    if agent.standalone:
        with open(get_qa_output_path(), 'w', encoding='utf-8') as f:
            for result, question in zip(notebooks, all_questions):
                result_json = {
                    "custom_id": question["question_id"],
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": result.get_notes()
                                    }
                                }
                            ]
                        }
                    }
                }
                r = json.dumps(result_json)
                f.write(r + '\n')

        return None

    guard_job(results, model, stop)

    return queue_batch_job([
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
                "temperature": job_args['temperature'] if supports_temperature_param(model) else None,
                "frequency_penalty": job_args['frequency_penalty'],
                "presence_penalty": job_args['presence_penalty'],
                "max_completion_tokens": int(get_max_output_tokens(model))
            },
        }
        for _, question_set in questions.items()
        for question in question_set
    ])
