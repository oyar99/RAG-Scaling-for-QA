"""BM25 RAG system for document retrieval using BM25L algorithm."""

from rank_bm25 import BM25L
from logger.logger import Logger
from models.retrieved_result import RetrievedResult
from utils.tokenizer import tokenize


class BM25RAG:
    """BM25 RAG system for document retrieval using BM25L algorithm."""

    def __init__(self):
        self._index = None
        self._corpus = None

    def index(self, docs: list[str]) -> None:
        """Index the documents using BM25L algorithm.

        Args:
            docs (list[str]): List of documents to index.
        """
        Logger().info("Indexing documents using BM25L algorithm")
        # Tokenize the documents
        tokenized_docs = [tokenize(doc, ngrams=2, remove_stopwords=True) for doc in docs]

        # Index the documents using BM25L
        self._index = BM25L(tokenized_docs)
        self._corpus = docs

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedResult]:
        """Retrieve the top k documents for the given query.

        Args:
            query (str): The query string.
            k (int, optional): The number of top documents to retrieve. Defaults to 5.

        Returns:
            list[tuple[int, float]]: List of tuples containing document index and score.
        """
        Logger().info(f"Retrieving top {k} documents for query: {query}")
        # Tokenize the query
        tokenized_query = tokenize(query)

        # Get scores for the query
        scores = self._index.get_scores(tokenized_query)

        # Get top k documents with their scores
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        retrieved_docs = [RetrievedResult(
            corpus_id=idx, content=self._corpus[idx], score=score) for idx, score in top_k]

        Logger().info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
