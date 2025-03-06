"""Hotpot dataset module."""

import json
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.question_utils import filter_questions


# pylint: disable-next=too-few-public-methods
class Hotpot(Dataset):
    """Hotpot dataset class."""

    def __init__(self, args):
        super().__init__(args)
        Logger().info("Initialized an instance of the Hotpot dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the Hotpot dataset.

        Returns:
            list[DatasetSample]: the dataset samples
        """
        Logger().info("Reading the Hotpot dataset")
        conversation_id = self._args.conversation

        # pylint: disable=duplicate-code
        with open("datasets\\hotpot\\hotpot_dev_distractor_v1.json", encoding="utf-8") as hotpot_dataset:
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
                            if sample['type'] == 'bridge' else QuestionCategory.OPEN_DOMAIN
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(hotpot_dataset)
                if conversation_id is None or sample['_id'] == conversation_id
            ]
            super().process_dataset(dataset)
            Logger().info(
                f"Hotpot dataset read successfully. Total samples: {len(dataset)}")

            return dataset
        # pylint: enable=duplicate-code
