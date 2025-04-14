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

    def __init__(
        self,
        args,
        name=None,
    ):
        self._args = args
        self._dataset = None
        self._dataset_map = None
        self._prompt_dict = {
            'qa_rel': (QA_PROMPT_RELEVANT if (args.model is not None) and
                       args.model in ('o3-mini', 'gpt-4o-mini', 'gpt-4o-mini-batch') else QA_PROMPT_RELEVANT_EXPLICIT),
            'qa_all': QA_PROMPT_ALL,
        }

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

    def get_prompt(self, prompt_id: str) -> str:
        """
        Returns the prompt builder.

        Args:
            prompt_id (str): the prompt id to retrieve
        Raises:
            ValueError: if the prompt id is not found in the prompt dictionary

        Returns:
            str: the prompt builder
        """
        if prompt_id not in self._prompt_dict:
            Logger().error(
                f"Prompt id {prompt_id} not found in the prompt dictionary.")
            raise ValueError(
                f"Prompt id {prompt_id} not found in the prompt dictionary.")

        return self._prompt_dict[prompt_id]

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


QA_PROMPT_RELEVANT = '''You are a helpful Question Answering assistant. You will be presented with relevant \
passages, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the passages when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. If there is no enough information in the passages to answer the question, please answer "N/A".

For example if the question is:

Q: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

Your answer should be: "No"

Below are the passages.

{context}
'''

QA_PROMPT_RELEVANT_EXPLICIT = '''You are a helpful Question Answering assistant. You will be presented with relevant \
passages, followed by a question. Your task is to provide an EXACT answer, using only words found in the passages when \
possible. UNDER NO CIRCUMSTANCES should you include any additional commentary, explanations, reasoning, or notes in \
your response. Your response MUST be concise and to the point.

Below is an example of given passages, a question, and the expected answer.

Passages:

Universal Pictures: Universal owned the rights to the \"Oswald the Lucky Rabbit\" character, although Walt Disney and \
Ub Iwerks had created Oswald, and their films had enjoyed a successful theatrical run. After Charles Mintz had unsuccessfully \
demanded that Disney accept a lower fee for producing the property, Mintz produced the films with his own group of animators. \
Instead, Disney and Iwerks created Mickey Mouse who in 1928 starred in the first \"sync\" sound animated short, Steamboat Willie. \
This moment effectively launched Walt Disney Studios' foothold, while Universal became a minor player in film animation. \
Universal subsequently severed its link to Mintz and formed its own in-house animation studio to produce Oswald cartoons \
headed by Walter Lantz.

The Mickey Mouse Club: The Mickey Mouse Club is an American variety television show that aired intermittently from 1955 to \
1996 and returned in 2017 to social media. Created by Walt Disney and produced by Walt Disney Productions, the program was \
first televised in 1955 by ABC, featuring a regular but ever-changing cast of mostly teen performers. ABC broadcast reruns \
weekday afternoons during the 1958 -- 1959 season, airing right after American Bandstand. The show was revived after its \
initial 1955 -- 1959 run on ABC, first from 1977 -- 1979 for first-run syndication, again from 1989 -- 1996 as The All-New \
Mickey Mouse Club (also known to fans as MMC from 1993 -- 1996) airing exclusively on cable television's The Disney Channel, \
then rebooted in 2017 with the moniker Club Mickey Mouse airing exclusively on internet social media.

Question:
"What was the old show that was named after a character that Walt Disney created in 1928 called?"

Answer:
"The Mickey Mouse Club"

Your response MUST be formatted as a single line of text, containing ONLY the answer to the question. \
If the answer is not present or cannot be inferred with the information found in the passages, you MUST then respond with "N/A". 

DO NOT include any additional commentary, explanations, or reasoning in your response. For example, \
refrain from including notes like "(Note: Based on the information provided, the answer is...)"

Below are the relevant passages.

Passages:

{context}

-----

Remember to provide the answer in a single line, without any additional commentary, explanations, or extraneous text.
'''

QA_PROMPT_ALL = '''You are a helpful Question Answering assistant. You will be presented with all the passages \
in the dataset which may or may not be relevant to answer the given questions. Your task is to provide an EXACT \
answer, using only words found in the passages when possible. If the answer can be a single word \
(e.g., Yes, No, a date, or an object), please provide just that word. Note that all questions are answerable \
with the provided passages, so reiterate if you do not find relevant information.

Questions are formatted as follows:

Q (<question_id>): "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"

Format your answer as a JSON object where each question is answered exactly once. Your response should also honor the given question \
order and question ids.

Below are the passages in the dataset. (The passages may be truncated due to length constraints)

{context}
'''
