"""Tokenizer module."""
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk


def normalize(
    text: str,
    is_remove_stopwords: bool = False,
    use_stemer: bool = False
) -> str:
    """
    Normalizes the input text by lowercasing, removing punctuation, articles, extra whitespace, and 
    optionally removing stopwords and applying stemming.

    Adapted from the MRQA-Shared-Task-2019

    See https://mrqa.github.io/

    Args:
        text (str): The input text to normalize.

    Returns:
        normalized_text (str): The normalized text.
    """
    def remove_articles(text: str):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def join_empty_space(text: str):
        return " ".join(text.split())

    def remove_punc(text: str):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str):
        return text.lower()

    def remove_stopwords(text: str):
        if not is_remove_stopwords:
            return text
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        return " ".join(word for word in text.split() if word not in stop_words)

    def stem(text: str):
        if not use_stemer:
            return text
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab')
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return " ".join(stemmer.stem(word) for word in word_tokenize(text))

    return stem(remove_stopwords(join_empty_space(remove_articles(remove_punc(lower(text))))))


def tokenize(
    text: str,
    ngrams: int = 1,
    remove_stopwords: bool = False,
    use_stemmer: bool = False
) -> list[str]:
    """
    Tokenizes the input text into a list of tokens.

    Args:
        text (str): The input text to tokenize.
        ngrams (int): The size of the ngrams to generate. Default is 1 and the maximum is 5.
        remove_stopwords (bool): Whether to remove stopwords. Default is False.
        use_stemmer (bool): Whether to apply stemming. Default is False.

    Returns:
        tokens (list[str]): A list of tokens.
    """
    unigrams = normalize(
        text, is_remove_stopwords=remove_stopwords, use_stemer=use_stemmer).split()

    all_ngrams = [
        " ".join(unigrams[i:i + n])
        for n in range(1, min(ngrams + 1, 6))
        for i in range(len(unigrams) - n + 1)
    ]

    return all_ngrams
