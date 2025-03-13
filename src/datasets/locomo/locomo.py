"""Locomo dataset module."""

import json
import re
from typing import Optional, override

from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.question_utils import filter_questions


class Locomo(Dataset):
    """Locomo dataset class."""

    QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
conversation between two users, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the conversations when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. For example if the question is:

Q: "what book did Carlos buy on his birthday?"

Your answer should be: "Becoming Nicole"

The conversation takes place over multiple days and the date of each conversation is appended at the end of each message.

Below are the relevant messages in the conversation.

{context}
'''

    def __init__(self, args):
        super().__init__(args)
        Logger().info("Initialized an instance of the Locomo dataset")

    def read(self) -> list[DatasetSample]:
        """
        Reads the Locomo dataset.

        Returns:
            dataset (list[DatasetSample]): the Locomo dataset
        """
        Logger().info("Reading Locomo dataset")
        conversation_id = self._args.conversation
        with open("datasets\\locomo\\locomo10.json", "r", encoding="utf-8") as locomo_dataset:
            dataset = [
                DatasetSample(
                    sample_id=conversation_sample['sample_id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(
                                doc_id=ev,
                                content=conversation_sample['conversation'][f"session_{ev.split(':')[0][1:]}"][int(
                                    ev.split(':')[1]) - 1]['text'])
                                  for ev in qa['evidence']
                                  if f"session_{ev.split(':')[0][1:]}" in conversation_sample['conversation'] and int(
                                      ev.split(':')[1]) - 1 <
                                  len(conversation_sample['conversation'][f"session_{ev.split(':')[0][1:]}"])],
                            question_id=f'{conversation_sample["sample_id"]}-{i + 1}',
                            question=qa['question'],
                            answer=[str(qa.get('answer')) or str(qa.get(
                                'adversarial_answer'))],
                            category=QuestionCategory(qa['category'])
                        ) for i, qa in enumerate(conversation_sample['qa'])], self._args.questions, self._args.category)
                    )
                )
                for conversation_sample in json.load(locomo_dataset)
                if conversation_id is None or conversation_sample['sample_id'] == conversation_id
            ]
            super().process_dataset(dataset)
            Logger().info(
                f"Locomo dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    def read_corpus(self) -> list[Document]:
        """
        Reads the LoCoMo dataset corpus.

        Returns:
            corpus (list[str]): list of docs (messages) from the corpus
        """
        Logger().info("Reading the LoCoMo dataset corpus")
        with open("datasets\\locomo\\locomo10.json", encoding="utf-8") as locomo_corpus:
            corpus = json.load(locomo_corpus)

            pattern = re.compile(r"^session_\d+$")
            conversation_id = self._args.conversation

            corpus = [
                Document(
                    doc_id=message['dia_id'],
                    content=f"At {conversation_sample['conversation'][f'{key}_date_time']} {message['speaker']} said: \
{message['text']}"
                )
                for conversation_sample in corpus[:self._args.limit]
                if conversation_id is None or conversation_sample['sample_id'] == conversation_id
                for key, session in conversation_sample['conversation'].items() if pattern.match(key)
                for message in session
            ]
            Logger().info(
                f"LoCoMo dataset corpus read successfully. Total documents: {len(corpus)}")

            return corpus

    @override
    def get_question(self, question_id: str) -> Optional[QuestionAnswer]:
        """
        Gets a question from the dataset.

        Args:
            question_id (str): the id of the question to be retrieved
            
        Raises:
            ValueError: if the dataset is not read or the question id is not found in the dataset

        Returns:
            question (Optional[QuestionAnswer]): the question if found, None otherwise
        """
        if not self._dataset:
            Logger().error("Dataset not read. Please read the dataset before getting questions.")
            raise ValueError(
                "Dataset not read. Please read the dataset before getting questions.")

        match = re.match(r'^(conv-.\d+)-(\d+)$',
                         question_id)
        sample_id = match.group(1)
        message_id = match.group(2)

        if sample_id not in self._dataset_map:
            Logger().error(
                f"Sample id {sample_id} not found in the dataset.")
            raise ValueError(
                f"Sample id {sample_id} not found in the dataset.")

        return self._dataset_map[sample_id]['qa'][int(message_id)-1]
