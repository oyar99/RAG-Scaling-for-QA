"""Musique dataset class."""

import json
from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.question_utils import filter_questions


class MuSiQue(Dataset):
    """MuSiQue dataset class"""

    def __init__(self, args):
        super().__init__(args)
        Logger().info("Initialized an instance of the MuSiQue dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the MuSiQue dataset.

        Returns:
            dataset (list[DatasetSample]): the dataset samples
        """
        Logger().info("Reading the MuSiQue dataset")
        conversation_id = self._args.conversation

        with open("datasets\\musique\\musique_dev.json", encoding="utf-8") as musique_dataset:
            dataset = [
                DatasetSample(
                    sample_id=sample['id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(doc_id=doc['paragraph_text'], content=doc['paragraph_text'])
                                  for doc in sample['paragraphs'] if doc['is_supporting']],
                            question_id=sample['id'],
                            question=sample['question'],
                            answer=[str(sample['answer'])] + sample['answer_aliases'],
                            category=QuestionCategory.MULTI_HOP
                        )], self._args.questions, self._args.category)
                    )
                )
                for sample in json.load(musique_dataset)
                if conversation_id is None or sample['id'] == conversation_id
            ]
            dataset = super().process_dataset(dataset)
            Logger().info(
                f"MuSiQue dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    def read_corpus(self) -> list[str]:
        """
        Reads the MuSiQue dataset and returns the corpus.

        Returns:
            corpus (list[str]): the corpus
        """
        Logger().info("Reading the MuSiQue dataset corpus")
        with open("datasets\\musique\\musique_corpus.json", encoding="utf-8") as musique_corpus:
            corpus = json.load(musique_corpus)
            corpus = [
                Document(doc_id=doc['text'], content=doc['text'])
                for doc in corpus
            ]
            Logger().info(
                f"MuSiQue dataset corpus read successfully. Total documents: {len(corpus)}")

            return corpus
