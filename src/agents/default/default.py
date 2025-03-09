"""Default system for document retrieval that chooses docs directly from the dataset."""

from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult


class Default(Agent):
    """Default System"""

    def __init__(self, args):
        super().__init__(args)
        self._qa_prompt = None

    def index(self, dataset: Dataset) -> None:
        """Index the documents

        Args:
            dataset (Dataset): The dataset to index
        """
        corpus = dataset.read_corpus()
        self._corpus = corpus
        self._index = corpus
        self._qa_prompt = dataset.QA_PROMPT

    def reason(self, _: str) -> NoteBook:
        """Dummy reasoning since it just returns all the documents in the dataset.

        Args:
            _ (str): the question to reason about

        Raises:
            ValueError: if the index is not created

        Returns:
            NoteBook: the notebook containing the documents
        """
        # pylint: disable=duplicate-code
        if not self._index or not self._corpus:
            Logger().error("Index not created. Please index the dataset before retrieving documents.")
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")
        # pylint: enable=duplicate-code

        notebook = NoteBook()

        notes = self._qa_prompt.format(
            context='\n'.join(doc['content'] for doc in self._index))

        notebook.update_notes(notes)
        notebook.update_sources([RetrievedResult(
            doc_id=doc['doc_id'], content=doc['content'], score=None)
            for doc in self._index])

        return notebook
