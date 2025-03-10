"""Utilities for tokenization and token counting."""
import tiktoken

from logger.logger import Logger


def estimate_num_tokens(prompt: str, model: str) -> int:
    """
    Estimate the number of tokens in a prompt for a given model.

    Args:
        prompt (str): the prompt to be estimated
        model (str): the model to be used for estimation

    Returns:
        estimate_tokens (int): an estimate of the number of tokens in the prompt when using the given model
    """
    encoding = tiktoken.encoding_for_model(model)
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
        'gpt-35-turbo': 0.25,
    }

    if model not in cost_per_million_tokens:
        Logger().warn(f"Model {model} not found in cost database.")
        return 0.0

    cost_per_token = cost_per_million_tokens[model] / 1_000_000
    cost = num_tokens * cost_per_token
    return cost
