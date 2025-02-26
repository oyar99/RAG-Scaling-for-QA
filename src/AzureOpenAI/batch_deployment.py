from AzureOpenAI.openai_client import get_openai_client
from models.question import Question
from openai import AzureOpenAI
import json


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
        max_tokens (int, optional): the maximum number of tokens that can be generated in the completion. Defaults to 1000.
        frequency_penalty (float, optional): penalize new tokens based on their existing frequency in the text so far. Defaults to 0.0.
        presence_penalty (float, optional): penalize new tokens based on whether they appear in the text so far. Defaults to 0.0.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
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
    
    # TODO: Send requests to Azure OpenAI

    openai_client = get_openai_client()
