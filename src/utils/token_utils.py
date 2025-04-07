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

# pylint: disable-next=too-many-locals
def search_best_interval(
    content: str,
    must_have_texts: list[str],
    context_starts_idx: int,
    max_tokens: int,
    encoding: tiktoken.Encoding
) -> tuple[int, int]:
    """
    Find the largest interval (start, end) such that the length of encoded(content)[start, end] is not greater 
    than max_tokens and content is guaranteed to contain all substrings in must_have_texts.
    """
    must_have_indices = [
        (content.index(text), content.index(text) + len(text))
        for text in must_have_texts
    ]

    smallest_start = min(start for start, _ in must_have_indices)
    largest_end = max(end for _, end in must_have_indices)

    def can_extend(start_idx: int, end_idx: int) -> tuple[bool, bool]:
        token_count = len(encoding.encode(
            content[:context_starts_idx] + content[start_idx:end_idx]))
        return (token_count <= max_tokens, token_count == max_tokens)

    max_reached = False

    # Binary search to find the maximum extension to the left
    best_left = smallest_start
    left_start, left_end = context_starts_idx, smallest_start
    while left_start <= left_end:
        mid = (left_start + left_end + 1) // 2
        can_extend_flag, max_r = can_extend(mid, largest_end)
        if not can_extend_flag:
            left_start = mid + 1
        else:
            left_end = mid - 1
            best_left = mid

        if max_r:
            max_reached = True
            break

    best_right = largest_end

    if not max_reached:
        # Binary search to find the maximum extension to the right
        right_start, right_end = largest_end, len(content)
        while right_start <= right_end:
            mid = (right_start + right_end) // 2
            can_extend_flag, max_r = can_extend(best_left, mid)
            if can_extend_flag:
                best_right = mid
                right_start = mid + 1
            else:
                right_end = mid - 1

            if max_r:
                max_reached = True
                break

    return (best_left, best_right)


def truncate_content(content: str, must_have_texts: list[str], context_starts_idx: int, model: str) -> str:
    """
    Truncates the content to fit within the specified token limit.

    Args:
        content (str): the content to be truncated
        must_have_texts (list[str]): list of docs that need not to be truncated
        context_starts_idx (int): the index where the actual context that can be truncated starts
        model (str): the model to be used for tokenization

    Returns:
        truncated_content (str): the truncated content
    """
    encoding = get_encoding(model)
    tokens = encoding.encode(content)

    max_tokens_map = {
        'gpt-4o-mini': 128_000 * 0.88,
        'o3-mini': 200_000 * 0.88,
    }

    max_tokens = int(max_tokens_map.get(model, 0))

    if len(tokens) > max_tokens:
        Logger().warn(
            f"Content exceeds the maximum token limit of {max_tokens}. Truncating content..."
        )

        start, end = search_best_interval(
            content, must_have_texts, context_starts_idx, max_tokens, encoding)

        content = content[:context_starts_idx] + content[start: end]
        encoded_content = encoding.encode(content)

        if len(encoded_content) > max_tokens:
            Logger().warn("Truncation failed, content still exceeds max tokens. Proceeding with naive truncation")
            content = encoding.decode(encoded_content[:max_tokens])

        Logger().debug(f"Truncating content: {start}-{end}")
        Logger().debug(f"token length: {len(encoded_content)}")

    return content


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
        'o3-mini': 200_000 * 0.1,
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
        'o3-mini': 0.55,
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
