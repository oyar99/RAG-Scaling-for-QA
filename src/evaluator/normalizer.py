"""
Utility functions for normalizing text taken from the MRQA-Shared-Task-2019

See https://mrqa.github.io/
"""

import re
import string


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Args:
        s (str): the text to be normalized

    Returns:
        str: the normalized text
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def join_empty_space(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return join_empty_space(remove_articles(remove_punc(lower(s))))
