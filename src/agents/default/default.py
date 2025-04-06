"""Default system for document retrieval that chooses docs directly from the dataset."""

from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult


class Default(Agent):
    """Default System"""

    def __init__(self, args):
        self._corpus = None
        self._index = None
        self._qa_prompt = None
        super().__init__(args)

        # Support batch reasoning
        self.support_batch = True

    def index(self, dataset: Dataset) -> None:
        """Index the documents

        Args:
            dataset (Dataset): The dataset to index
        """
        corpus = dataset.read_corpus()
        self._corpus = corpus
        self._index = corpus
        self._qa_prompt = dataset.get_prompt('qa_all')

    def batch_reason(self, _: list[QuestionAnswer]) -> NoteBook:
        """
        Dummy batch reasoning since it just returns all the documents in the dataset.

        Args:
            questions (QuestionAnswer): list of questions to reason about
        """
        if not self._index or not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")

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
            else:
                formatted_docs.append("Passages:")
            formatted_docs.extend(doc['content'] for doc in docs)

        notes = self._qa_prompt.format(
            context='\n'.join(formatted_docs)
        )
        notebook.update_notes(notes)
        notebook.update_actual_context_idx(notes.index(
            'sample_id') if 'sample_id' in notes else notes.index('Passages:'))
        notebook.update_sources([RetrievedResult(
            doc_id=doc['doc_id'], content=doc['content'], score=None)
            for doc in self._index])
        return notebook

    def reason(self, _: str) -> NoteBook:
        """Dummy reasoning since it just returns all the documents in the dataset.

        Args:
            _ (str): the question to reason about

        Raises:
            ValueError: if the index is not created

        Returns:
            notebook (NoteBook): the notebook containing the documents
        """
        raise NotImplementedError(
            "Default agent does not support single question reasoning. Use batch_reason instead."
        )
