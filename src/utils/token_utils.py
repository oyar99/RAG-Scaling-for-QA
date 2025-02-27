"""Utilities for tokenization and token counting."""
import tiktoken

def estimate_num_tokens(prompt: str, model: str) -> int:
    """Estimate the number of tokens in a prompt for a given model.

    Args:
        prompt (str): the prompt to be tokenized
        model (str): the model to be used for tokenization

    Returns:
        int: an estimate of the number of tokens in the prompt when using the given model
    """
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens
