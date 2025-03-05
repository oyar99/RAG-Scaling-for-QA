"""Hotpot dataset module."""

import json
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.question_utils import filter_questions

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with relevant \
passages, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the passages when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. For example if the question is:

Q: "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?"

Your answer should be: "No"

Below are the passages.

{passages}
'''

"""
TODO: Right now this is pretty much the same implementation as HotpotQA. However, 
I don't want to merge them just yet because I might have to use the additional evidences provided by 2wiki later on. 
See https://github.com/oyar99/HybridLongMemGPT/issues/9
"""


class TwoWiki(Dataset):
    """Hotpot dataset class."""

    def __init__(self, args):
        self._args = args
        self._dataset = None
        self._dataset_map = None
        super().__init__()
        Logger().info("Initialized an instance of the 2Wiki dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the Hotpot dataset.

        Returns:
            list[DatasetSample]: the dataset samples
        """
        Logger().info("Reading the 2WikiMultihopQA dataset")
        conversation_id = self._args.conversation

        with open("datasets\\twowikimultihopqa\\dev.json", encoding="utf-8") as two_wiki_dataset:
            dataset = [
                DatasetSample(
                    sample_id=sample['_id'],
                    sample=DatasetSampleInstance(
                        context=([' '.join(doc[1])
                                 for doc in sample['context']]),
                        qa=filter_questions([QuestionAnswer(
                            question_id=sample['_id'],
                            question=sample['question'],
                            answer=sample['answer'],
                            category=QuestionCategory.MULTI_HOP
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(two_wiki_dataset)
                if conversation_id is None or sample['_id'] == conversation_id
            ]
            dataset = [sample for sample in dataset if len(
                sample['sample']['qa']) > 0][:self._args.limit]

            self._dataset = dataset
            self._dataset_map = {
                sample['sample_id']: sample['sample']
                for sample in dataset
            }

            Logger().info(
                f"2Wiki dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    def build_system_prompt(self) -> dict[str, str]:
        """
        Builds the system prompt for the Hotpot dataset.

        Returns:
            dict[str, str]: the system prompt
        """
        if not self._dataset:
            Logger().error("Dataset not read. Please read the dataset before building system prompts.")
            raise ValueError(
                "Dataset not read. Please read the dataset before building system prompts.")

        Logger().info("Building system prompts")

        system_prompt = {
            sample['sample_id']: QA_PROMPT.format(
                passages='\n'.join(sample['sample']['context']))
            for sample in self._dataset
        }

        Logger().info("System prompts built successfully")

        return system_prompt
