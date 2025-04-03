"""Oracle agent that answers questions using the truth supporting docs."""
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult


class Oracle(Agent):
    """
    An oracle agent that answers questions using the truth supporting docs.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """
        Cheats by looking at the questions beforehand and indexing them.

        Args:
            dataset (Dataset): The dataset to index.
        """
        corpus = dataset.read_corpus()

        questions = [q for questions in dataset.get_questions().values()
                     for q in questions]

        self._index = {
            question['question']: question
            for question in questions
        }

        self._qa_prompt = dataset.get_prompt('qa_rel')
        self._corpus = corpus

    def reason(self, question: str) -> NoteBook:
        """
        Uses its question index to answer the question.
        If the question is not in the index, it raises a ValueError.

        Args:
            question (str): The question to answer.

        Returns:
            NoteBook: The notebook containing the answer and sources.

        Raises:
            ValueError: If the question has no answer.
        """
        if not self._index:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")

        question_obj = self._index.get(question)

        if question_obj is None:
            raise ValueError(f"Question {question} has no answer.")

        docs = question_obj.get('docs')

        notebook = NoteBook()

        notes = self._qa_prompt.format(
            context='\n'.join(doc['content'] for doc in docs))

        notebook.update_notes(notes)
        notebook.update_sources([RetrievedResult(
            doc_id=doc['doc_id'], content=doc['content'], score=100.0)
            for doc in docs])

        return notebook

    def batch_reason(self, _: list[str]) -> NoteBook:
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the Oracle agent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the Oracle agent.")
