"""Default system for document retrieval that chooses docs directly from the dataset."""

from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult


class Default(Agent):
    """Default System"""

    def __init__(self, args):
        self._corpus = None
        self._index = None
        self._qa_prompt = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """Index the documents

        Args:
            dataset (Dataset): The dataset to index
        """
        corpus = dataset.read_corpus()
        self._corpus = corpus
        self._index = corpus
        self._qa_prompt = dataset.get_prompt('qa_all')

    def reason(self, _: str) -> NoteBook:
        """Dummy reasoning since it just returns all the documents in the dataset.

        Args:
            _ (str): the question to reason about

        Raises:
            ValueError: if the index is not created

        Returns:
            notebook (NoteBook): the notebook containing the documents
        """
        # pylint: disable=duplicate-code
        if not self._index or not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")
        # pylint: enable=duplicate-code

        notebook = NoteBook()

        grouped_docs = {}
        for doc in self._index:
            folder_id = doc.get('folder_id')
            if folder_id:
                grouped_docs.setdefault(folder_id, []).append(doc)
            else:
                grouped_docs.setdefault(None, []).append(doc)

        formatted_docs = []
        for folder_id, docs in grouped_docs.items():
            if folder_id is not None:
                formatted_docs.append(f"sample_id: {folder_id}")
            formatted_docs.extend(doc['content'] for doc in docs)

        notebook.update_notes('\n'.join(formatted_docs))
        notebook.update_sources([RetrievedResult(
            doc_id=doc['doc_id'], content=doc['content'], score=None)
            for doc in self._index])

        return notebook
