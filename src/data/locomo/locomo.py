"""Locomo dataset module."""

import json
import os
import re
from typing import Optional

from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.document import Document
from models.question_answer import QuestionAnswer, QuestionCategory
from utils.hash_utils import get_content_hash
from utils.question_utils import filter_questions


def session_id(doc_id: str) -> str:
    """
    Extracts the session id from the doc_id.

    Args:
        doc_id (str): the document id

    Returns:
        session_id (str): the session id
    """
    return f"session_{doc_id.split(':')[0][1:]}"


def dia_idx(doc_id: str) -> int:
    """
    Extracts the index of the dialogue from the doc_id.

    Args:
        doc_id (str): the document id

    Returns:
        dia-idx (int): the index of the dialogue
    """
    return int(doc_id.split(':')[1]) - 1


def format_content(date: str, message: int, speaker: str, text: str, alt_text: str = None) -> str:
    """
    Formats the content of a message.

    Args:
        date (str): the date of the message
        message (int): the message number
        speaker (str): the speaker of the message
        text (str): the text of the message
        alt_text (str): the alt text of attached images if any
    """
    return f"At around {date}, during message {message}, {speaker} said: {text}" if alt_text is None else \
        f"At around {date}, during message {message}, {speaker} said: {text} - Attached image: {alt_text}"


class Locomo(Dataset):
    """Locomo dataset class."""

    QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with snippets from a \
conversation between two users, followed by a question. Your task is to provide an EXACT and short answer, using words \
found in the conversations when possible. If the answer can be a single word (e.g., Yes, No, or an entity), please \
answer with just that word. For dates, always answer with ABSOLUTE dates such as "5 July 2023" or "week before 5 June" instead \
of relative answers such as "Yesterday" or "last week" since your answers should not depend on the current date.

For example, given the following conversation:

"At around 1:50 pm on 17 August, 2023, during message 15, Caroline said: I'm always here for you, Mel! We had a blast last year \
at the Pride fest. Those supportive friends definitely make everything worth it!"

And given the following question:

Q: "When did Caroline and Melanie go to a pride festival together?"

Your answer should be: 

"2022"

The conversation takes place over multiple days and the date of each conversation is added at the beginning of each message.

Below are the relevant messages in the conversation.

{context}
'''

    def __init__(self, args):
        super().__init__(args, name="locomo")
        Logger().info("Initialized an instance of the Locomo dataset")

    # @override
    def read(self) -> list[DatasetSample]:
        """
        Reads the Locomo dataset.

        Returns:
            dataset (list[DatasetSample]): the Locomo dataset
        """
        Logger().info("Reading Locomo dataset")
        conversation_id = self._args.conversation
        file_path = os.path.join("data", "locomo", "locomo10.json")

        with open(file_path, "r", encoding="utf-8") as locomo_dataset:
            dataset = [
                DatasetSample(
                    sample_id=cs['sample_id'],
                    sample=DatasetSampleInstance(
                        qa=filter_questions([QuestionAnswer(
                            docs=[Document(
                                doc_id=ev,
                                content=format_content(
                                    date=cs['conversation'][f'{session_id(ev)}_date_time'],
                                    message=dia_idx(ev) + 1,
                                    speaker=cs['conversation'][session_id(
                                        ev)][dia_idx(ev)]['speaker'],
                                    text=cs['conversation'][session_id(
                                        ev)][dia_idx(ev)]['text'],
                                    alt_text=cs['conversation'][session_id(ev)][dia_idx(ev)].get('blip_caption'))
                            )
                                for ev in qa['evidence']
                                if session_id(ev) in cs['conversation'] and dia_idx(ev) <
                                len(cs['conversation'][session_id(ev)])],
                            question_id=f'{cs["sample_id"]}-{get_content_hash(qa["question"])}',
                            question=qa['question'],
                            answer=[str(qa.get('answer')) or str(qa.get(
                                'adversarial_answer'))],
                            category=QuestionCategory(qa['category'])
                        ) for _, qa in enumerate(cs['qa'])], self._args.questions, self._args.category)
                    )
                )
                for cs in json.load(locomo_dataset)
                if conversation_id is None or cs['sample_id'] == conversation_id
            ]
            dataset = super().process_dataset(dataset)

            Logger().info(
                f"Locomo dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    # @override
    def read_corpus(self) -> list[Document]:
        """
        Reads the LoCoMo dataset corpus.

        Returns:
            corpus (list[str]): list of docs (messages) from the corpus
        """
        Logger().info("Reading the LoCoMo dataset corpus")
        file_path = os.path.join("data", "locomo", "locomo10.json")
        with open(file_path, encoding="utf-8") as locomo_corpus:
            corpus = json.load(locomo_corpus)

            pattern = re.compile(r"^session_\d+$")
            conversation_id = self._args.conversation

            corpus = [
                Document(
                    doc_id=message['dia_id'],
                    content=format_content(
                        date=conversation_sample['conversation'][f'{key}_date_time'],
                        message=dia_idx(message['dia_id']) + 1,
                        speaker=message['speaker'],
                        text=message['text'],
                        alt_text=message.get('blip_caption')
                    ),
                )
                for conversation_sample in corpus[:self._args.limit]
                if conversation_id is None or conversation_sample['sample_id'] == conversation_id
                for key, session in conversation_sample['conversation'].items() if pattern.match(key)
                for message in session
            ]
            super()._log_dataset_stats(corpus)

            return corpus

    # @override
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

        match = re.match(r'^(conv-.\d+)-(.*)$',
                         question_id)
        sample_id = match.group(1)
        message_id = match.group(2)

        if sample_id not in self._dataset_map:
            Logger().error(
                f"Sample id {sample_id} not found in the dataset.")
            raise ValueError(
                f"Sample id {sample_id} not found in the dataset.")

        return next((
            qa for qa in self._dataset_map[sample_id]['qa']
            if get_content_hash(qa['question']) == message_id), None
        )
