"""A module to create a dataset class."""

from abc import ABC, abstractmethod

from logger.logger import Logger
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
    QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with relevant \
passages, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the passages when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. For example if the question is:

Q: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

Your answer should be: "No"

Below are the passages.

{context}
'''

    def __init__(self, args):
        self._args = args
        self._dataset = None
        self._dataset_map = None

    @abstractmethod
    def read(self) -> list[DatasetSample]:
        """Reads a dataset and converts it to a list of DatasetSample instances.

        Args:
            args (Namespace): the arguments passed to the script

        Returns:
            list[DatasetSample]: the dataset as a list of DatasetSample instances
        """

    def process_dataset(self, dataset: list[DatasetSample]) -> None:
        """Saves the dataset and its map for quick question retrieval in memory

        Args:
            dataset (list[DatasetSample]): the dataset to be saved
        """
        dataset = [sample for sample in dataset if len(
            sample['sample']['qa']) > 0][:self._args.limit]

        self._dataset = dataset
        self._dataset_map = {
            sample['sample_id']: sample['sample']
            for sample in dataset
        }

    def get_question(self, question_id: str) -> QuestionAnswer:
        """
        Gets a question from the dataset.

        Args:
            question_id (str): the unique identifier of the question

        Returns:
            QuestionAnswer: the question
        """
        if not self._dataset_map:
            Logger().error("Dataset not read. Please read the dataset before getting questions.")
            raise ValueError(
                "Dataset not read. Please read the dataset before getting questions.")

        if question_id not in self._dataset_map:
            Logger().error(
                f"Question id {question_id} not found in the dataset.")
            raise ValueError(
                f"Question id {question_id} not found in the dataset.")

        # Question_id is the same as the sample_id in this dataset
        return next((qa for qa in self._dataset_map[question_id]['qa'] if qa['question_id'] == question_id), None)

    def get_questions(self) -> dict[str, list[QuestionAnswer]]:
        """Get the questions from the Hotpot dataset.

        Returns:
            dict[str, list[QuestionAnswer]]: the questions
        """
        if not self._dataset:
            Logger().error("Dataset not read. Please read the dataset before getting questions.")
            raise ValueError(
                "Dataset not read. Please read the dataset before getting questions.")

        questions = {
            sample['sample_id']: sample['sample']['qa']
            for sample in self._dataset
        }

        total_questions = sum(len(qas) for qas in questions.values())
        Logger().info(f"Total questions retrieved: {total_questions}")

        return questions

    def build_system_prompt(self) -> dict[str, str]:
        """
        Builds the system prompt for QA tasks.

        Returns:
            dict[str, str]: A dictionary where the key is a dataset sample instance and the value its system prompt
        """
        if not self._dataset:
            Logger().error("Dataset not read. Please read the dataset before building system prompts.")
            raise ValueError(
                "Dataset not read. Please read the dataset before building system prompts.")

        Logger().info("Building system prompts")

        system_prompt = {
            sample['sample_id']: self.QA_PROMPT.format(
                context='\n'.join(sample['sample']['context']))
            for sample in self._dataset
        }

        Logger().info("System prompts built successfully")

        return system_prompt
