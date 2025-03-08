"""An agent module"""

from abc import ABC, abstractmethod
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult


class NoteBook:
    """A notebook class for storing notes and any other bookeeping stuff the agent needs."""

    def __init__(self):
        self._sources = []
        self._notes = None

    def update_notes(self, notes: str) -> None:
        """Updates the notebook."""
        self._notes = notes

    def get_notes(self) -> str:
        """Returns the notebook."""
        return self._notes

    def update_sources(self, sources: list[RetrievedResult]) -> None:
        """Updates the sources."""
        self._sources = sources

    def get_sources(self) -> list[RetrievedResult]:
        """Returns the sources."""
        return self._sources


class Agent(ABC):
    """
    An abstract class representing a language agent that interacts with the dataset by:

        - Indexing all content into its memory storage mechanism. Different agents
        may implement different memory tiers to store semantic or episodic momories.

    Given a question, the agent can

        - Create a rationale-driven plan based on pre-defined actions to aid in decision-making
        - Execute the plan and determine if it has gathered sufficient resources to answer the question
        - Iterate on the plan as needed

    Finally the agent will return a detailed notebook with its findings for this question so another language
    agent can answers all questions in bulk.

    Args:
        ABC: an abstract base class
    """

    def __init__(self, args):
        self._args = args
        self._index = None
        self._corpus = None

    @abstractmethod
    def index(self, dataset: Dataset) -> None:
        """
        Indexes the contents of the given dataset

        Args:
            dataset (Dataset): The given dataset already initialized
        """

    @abstractmethod
    def reason(self, question: str) -> NoteBook:
        """
        Given a question, reasons about it using its index (memory) and returns a 
        detailed notebook (str) with its findings to generate a correct response.
        The agent should not respond to the question directly. Instead, it should create the notes with all its findings
        so that the response can easily be explainable.

        Args:
            question (str): the given question

        Returns:
            Notebook: the detailed findings to help answer this question (context)
        """
