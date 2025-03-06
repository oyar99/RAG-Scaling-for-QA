"""Musique dataset class"""

import json
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.question_utils import filter_questions


# pylint: disable-next=too-few-public-methods
class MuSiQue(Dataset):
    """MuSiQue dataset class"""

    def __init__(self, args):
        super().__init__(args)
        Logger().info("Initialized an instance of the MuSiQue dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the MuSiQue dataset.

        Returns:
            list[DatasetSample]: the dataset samples
        """
        Logger().info("Reading the MuSiQue dataset")
        conversation_id = self._args.conversation

        with open("datasets\\musique\\musique_dev.json", encoding="utf-8") as musique_dataset:
            dataset = [
                DatasetSample(
                    sample_id=sample['id'],
                    sample=DatasetSampleInstance(
                        context=([doc['paragraph_text']
                                 for doc in sample['paragraphs']]),
                        qa=filter_questions([QuestionAnswer(
                            question_id=sample['id'],
                            question=sample['question'],
                            answer=sample['answer'],
                            category=QuestionCategory.MULTI_HOP
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(musique_dataset)
                if conversation_id is None or sample['id'] == conversation_id
            ]
            super().process_dataset(dataset)
            Logger().info(
                f"MuSiQue dataset read successfully. Total samples: {len(dataset)}")

            return dataset
