"""Utilities for tokenization and token counting."""
import tiktoken

from logger.logger import Logger
from models.document import Document


def average_content_length(corpus: list[Document], model: str = None) -> tuple[float, float]:
    """
    Computes the average content length of the given documents measured in
    both characters and tokens

    Args:
        corpus (list[Document]): the given corpus
        model (str): the model identifier to use for tokenization

    Returns:
        float: tuple[float, float]: average number of characters and tokens in the corpus
    """
    avg_length = sum(len(doc['content']) for doc in corpus) / len(corpus)
    avg_tokens = 0.0

    if model is not None:
        avg_tokens = sum(estimate_num_tokens(
            doc['content'], model) for doc in corpus) / len(corpus)
    return avg_length, avg_tokens


def get_max_output_tokens(model: str) -> int:
    """
    Get the maximum number of output tokens for a given model.

    Args:
        model (str): the model to be used

    Returns:
        max_output_tokens (int): the maximum number of output tokens for the given model
    """
    max_tokens_map = {
        'gpt-4o-mini': 128_000 * 0.1,
        'gpt-4o-mini-batch': 128_000 * 0.1,
        'o3-mini': 200_000 * 0.1,
        'mistralai/Mistral-Nemo-Instruct-2407': 128_000 * 0.1,
        'Qwen/Qwen2.5-14B-Instruct': 32_000 * 0.1,
        'Qwen/Qwen2.5-1.5B-Instruct': 32_000 * 0.1,
        'google/gemma-3-12b-pt': 128_000 * 0.1,
    }

    return max_tokens_map.get(model, 0)


def estimate_num_tokens(prompt: str, model: str) -> int:
    """
    Estimate the number of tokens in a prompt for a given model.

    Args:
        prompt (str): the prompt to be estimated
        model (str): the model to be used for estimation

    Returns:
        estimate_tokens (int): an estimate of the number of tokens in the prompt when using the given model
    """
    encoding = get_encoding(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def estimate_cost(num_tokens: int, model: str) -> float:
    """
    Estimate the cost of generating a given number of tokens using a given model.
    If the model is not found in the cost database, a warning is logged and 0.0 is returned.

    Args:
        num_tokens (int): the number of tokens to be generated
        model (str): the model to be used for estimating cost

    Returns:
        estimate_cost (float): an estimate of the cost of generating the given number of tokens using the given model
    """
    cost_per_million_tokens = {
        'gpt-4o-mini': 0.075,
        'gpt-4o-mini-batch': 0.075,
        'o3-mini': 0.55,
        'mistralai/Mistral-Nemo-Instruct-2407': 0.01,
        'Qwen/Qwen2.5-14B-Instruct': 0.01,
        'Qwen/Qwen2.5-1.5B-Instruct': 0.01,
        'google/gemma-3-12b-pt': 0.01,
    }

    if model not in cost_per_million_tokens:
        Logger().warn(f"Model {model} not found in cost database.")
        return 0.0

    cost_per_token = cost_per_million_tokens[model] / 1_000_000
    cost = num_tokens * cost_per_token
    return cost


def get_encoding(model: str):
    """
    Get the encoding for a given model.

    Args:
        model (str): the model to be used

    Returns:
        encoding (tiktoken.Encoding): the encoding for the given model
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")
