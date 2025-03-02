"""A module to create a dataset class."""

from abc import ABC, abstractmethod

from models.question_answer import QuestionAnswer


class DatasetSampleInstance(dict):
    """A dataset sample instance is a representation of a QA problem instance.
    """

    def __init__(self, context: list[str], qa: list[QuestionAnswer]):
        dict.__init__(self, context=context, qa=qa)

    def __repr__(self):
        return f"""DatasetSampleInstance(context={self.get('context')}),\
    qa={self.get('qa')})"""


class DatasetSample(dict):
    """A dataset sample is a representation of a QA problem instance.

    A dataset sample is a dictionary with the following keys:

    - sample_id: the unique identifier of the sample
    - sample: a nested dictionary representing an instance of a QA problem
        - context: a list of docs (strings) representing the context of the QA problem
        - qa: a list of questions and answers
            - question: a string representing the question
            - answer: a string representing the answer
            - category: an integer representing the category of the question
            - question_id: a string representing the unique identifier of the question

    Args:
        dict: inherits from dict
    """

    def __init__(self, sample_id: str, sample: DatasetSampleInstance):
        dict.__init__(self, sample_id=sample_id,
                      sample=sample)

    def __repr__(self):
        return f"DatasetSample(sample_id={self.get('sample_id')}, sample={self.get('sample')})"


class Dataset(ABC):
    """An abstract class representing a dataset.

    Args:
        ABC: an abstract base class
    """

    @abstractmethod
    def read(self) -> list[DatasetSample]:
        """Reads a dataset and converts it to a list of DatasetSample instances.

        Args:
            args (Namespace): the arguments passed to the script

        Returns:
            list[DatasetSample]: the dataset as a list of DatasetSample instances
        """

    @abstractmethod
    def build_system_prompt(self) -> dict[str, str]:
        """Builds a system prompt for QA tasks

        Returns:
            dict[str, str]: A dictionary where the key is a dataset sample instance and the value its system prompt
        """

    @abstractmethod
    def get_questions(self) -> dict[str, list[QuestionAnswer]]:
        """Returns a list of questions from the dataset.

        Returns:
            list[QuestionAnswer]: the list of questions
        """
