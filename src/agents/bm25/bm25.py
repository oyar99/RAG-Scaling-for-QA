"""BM25 RAG system for document retrieval using BM25 algorithm."""

from rank_bm25 import BM25Okapi as BM25Ranker
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.document import Document
from models.retrieved_result import RetrievedResult
from utils.tokenizer import PreprocessingMethod, tokenize


def tokenize_doc(doc: Document) -> list[str]:
    """
    Tokenize the document.

    Args:
        doc (str): the document to tokenize

    Returns:
        _type_: _description_
    """
    return tokenize(
        doc['content'],
        ngrams=2,
        remove_stopwords=True,
        preprocessing_method=PreprocessingMethod.STEMMING
    )


class BM25(Agent):
    """BM25 RAG system for document retrieval using BM25 algorithm."""

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """Index the documents using BM25 algorithm.

        Args:
            dataset (Dataset): The dataset to index
        """
        Logger().info("Indexing documents using BM25 agent")
        corpus = dataset.read_corpus()

        # Index the documents using BM25
        self._index = BM25Ranker(
            corpus,
            tokenizer=tokenize_doc,
            b=0.75,
            k1=0.5
        )
        self._corpus = corpus
        self._qa_prompt = dataset.QA_PROMPT

        Logger().info("Successfully indexed documents")

    def reason(self, question: str) -> NoteBook:
        """
        Retrieve the top k documents for the given question.

        Args:
            question (str): The question

        Returns:
            notebook (NoteBook): The notebook containing the retrieved documents and notes gathered by the agent
        """
        # pylint: disable=duplicate-code
        if not self._index or not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")
        # pylint: enable=duplicate-code

        notebook = NoteBook()
        k = 5

        # Tokenize the query
        tokenized_query = tokenize(
            question,
            ngrams=2,
            remove_stopwords=True,
            preprocessing_method=PreprocessingMethod.STEMMING
        )

        # Get scores for the query
        scores = self._index.get_scores(tokenized_query)

        # Get top k documents with their scores
        top_k = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

        retrieved_docs = [RetrievedResult(
            doc_id=self._corpus[idx]['doc_id'],
            content=self._corpus[idx]['content'],
            score=score
        ) for idx, score in top_k]

        notebook.update_sources(retrieved_docs)

        # Update the notebook with the retrieved documents
        notes = self._qa_prompt.format(
            context='\n'.join(doc['content']
                              for doc in retrieved_docs))

        notebook.update_notes(notes)

        return notebook
