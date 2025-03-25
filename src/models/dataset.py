"""A module to create a dataset class."""

from abc import ABC, abstractmethod

from logger.logger import Logger
from models.document import Document
from models.question_answer import QuestionAnswer
from utils.token_utils import average_content_length


class DatasetSampleInstance(dict):
    """
    A dataset sample instance is a representation of a QA problem instance.

    Args:
        dict (Any): dictionary to store the QA problem instance
        qa (list[QuestionAnswer]): a list of questions and answers
    """

    def __init__(self, qa: list[QuestionAnswer]):
        dict.__init__(self, qa=qa)

    def __repr__(self):
        return f"""DatasetSampleInstance(qa={self.get('qa')}"""


class DatasetSample(dict):
    """
    A dataset sample is a representation of a QA problem instance.

    Args:
        dict (Any): inherits from dict
        sample_id (str): the unique identifier of the sample
        sample (DatasetSampleInstance): a nested dictionary representing an instance of a QA problem
    """

    def __init__(self, sample_id: str, sample: DatasetSampleInstance):
        dict.__init__(self, sample_id=sample_id,
                      sample=sample)

    def __repr__(self):
        return f"DatasetSample(sample_id={self.get('sample_id')}, sample={self.get('sample')})"


class Dataset(ABC):
    """
    An abstract class representing a dataset.

    Args:
        ABC: an abstract base class
    """
    QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with relevant \
passages, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the passages when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. If there is no enough information in the passages to answer the question, please answer "N/A".

For example if the question is:

Q: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

Your answer should be: "No"

Below are the passages.

{context}
'''

    def __init__(
        self,
        args,
        name=None,
    ):
        self._args = args
        self._dataset = None
        self._dataset_map = None

        self.name = name

    @abstractmethod
    def read(self) -> list[DatasetSample]:
        """
        Reads a dataset and converts it to a list of DatasetSample instances.

        Returns:
            dataset (list[DatasetSample]): the dataset as a list of DatasetSample instances
        """

    @abstractmethod
    def read_corpus(self) -> list[Document]:
        """
        Reads a dataset and converts it to a list of documents.

        Returns:
            corpus (list[Document]): a list of documents from the dataset
        """

    def process_dataset(self, dataset: list[DatasetSample]) -> list[DatasetSample]:
        """
        Creates a quick lookup table for the dataset and performs any necessary processing.

        Args:
            dataset (list[DatasetSample]): the dataset to use for processing
        """
        dataset = [sample for sample in dataset if len(
            sample['sample']['qa']) > 0][:self._args.limit]
        dataset = [
            {
                **sample,
                'sample': {
                    **sample['sample'],
                    'qa': [qa for qa in sample['sample']['qa'] if len(qa.get('docs', [])) > 0]
                }
            }
            for sample in dataset
        ]

        self._dataset = dataset
        self._dataset_map = {
            sample['sample_id']: sample['sample']
            for sample in dataset
        }

        return dataset

    def get_question(self, question_id: str) -> QuestionAnswer:
        """
        Gets a question from the dataset.

        Args:
            question_id (str): the unique identifier of the question to retrieve

        Raises:
            ValueError: if the dataset has not been read or the question id is not found in the dataset

        Returns:
            question (QuestionAnswer): the retrieved question
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

        return next((qa for qa in self._dataset_map[question_id]['qa'] if qa['question_id'] == question_id), None)

    def get_supporting_docs(self, question_id: str) -> list[Document]:
        """
        Gets the list of docs that support the given question

        Args:
            question_id (str): the unique identifier of the question for which to retrieve the supporting docs

        Raises:
            ValueError: if the dataset has not been read or the question id is not found in the dataset

        Returns:
            docs (list[Document]): list of docs that support the answer to the given question
        """
        question = self.get_question(question_id)

        return question.get('docs') if question else []

    def get_questions(self) -> dict[str, list[QuestionAnswer]]:
        """
        Get all questions from the dataset as a dictionary where the keys are the sample ids 
        and the values are lists of QuestionAnswer instances.

        Raises:
            ValueError: if the dataset has not been read

        Returns:
            questions (dict[str, list[QuestionAnswer]]): the questions
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

    def _log_dataset_stats(self, corpus: list[Document]) -> None:
        """
        Logs the dataset statistics.

        Args:
            corpus (list[Document]): the corpus to log statistics for
        """
        Logger().info(
            f"{self.name} dataset corpus stats. Total documents: {len(corpus)}")

        # Calculate the average document length
        avg_chars, avg_tokens = average_content_length(
            corpus, self._args.model)
        avg_tokens_str = f"{avg_tokens:.2f} tokens" if self._args.model else "unknown tokens"
        Logger().info(
            f"Average document length in the corpus: {avg_chars:.2f} characters ({avg_tokens_str})")
