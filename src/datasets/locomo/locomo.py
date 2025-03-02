"""Locomo dataset module."""

from collections import defaultdict
import json
import re

from logger.logger import Logger
from models.dataset import Dataset, DatasetSample, DatasetSampleInstance
from models.question_answer import QuestionAnswer
from utils.question_utils import filter_questions
from utils.token_utils import estimate_num_tokens

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
conversation between two users, followed by a question. Your task is to provide an EXACT answer, using only words \
found in the conversations when possible. If the answer can be a single word (e.g., Yes, No, a date, or an object), please \
provide just that word. For example if the question is:

Q: "what book did Carlos buy on his birthday?"

Your answer should be: "Becoming Nicole"

The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation:

Below is the conversation.

{conversation}
'''


def _parse_conversation(conversation, args) -> str:
    """Parses the conversation object in a human-readable format.

    Args:
        conversation (dict): the conversation object between two users

        args (Namespace): the arguments passed to the script

    Returns:
        str: a human readable string representation of the conversation
    """
    if args.execution == 'eval':
        Logger().warn("Context is not processed in evaluation mode")
        return ""

    pattern = re.compile(r"^session_\d+$")

    parsed_messages = [
        {'speaker': message['speaker'],
            'text': message['text'], 'session': key}
        for key, session in conversation.items() if pattern.match(key)
        for message in session
    ]

    grouped_messages = defaultdict(list)
    for message in parsed_messages:
        grouped_messages[message['session']].append(message)

    conversation = "\n".join(
        [
            f"DATE: {conversation[f'{session}_date_time']}\nCONVERSATION:\n" + "\n".join(
                [f"{message['speaker']} said: {message['text']}" for message in messages]
            )
            for session, messages in grouped_messages.items()
        ]
    )

    if args.dis_trunc:
        return conversation

    return _truncate_conversation(conversation, args.model)


def _truncate_conversation(conversation: str, model: str) -> str:
    """Truncates the conversation to 16,000 tokens.

    Args:
        conversation (str): the conversation to be truncated

        model (str): the model deployment identifier

    Returns:
        str: the truncated conversation
    """
    token_count = estimate_num_tokens(conversation, model)

    Logger().info(
        f"Conversation consists of approximately {token_count} tokens.")

    if token_count > 16000:
        Logger().warn("Conversation exceeds 16,000 tokens. Truncating..., This may affect the model's performance.")
        conversation = conversation[-16000:]

    return conversation


class Locomo(Dataset):
    """Locomo dataset class."""

    def __init__(self, args):
        self._args = args
        self._dataset = None
        super().__init__()
        Logger().info("Initialized an instance of the Locomo dataset")

    def read(self) -> list[DatasetSample]:
        """Reads the Locomo dataset.

        Returns:
            list[DatasetSample]: the Locomo dataset
        """
        Logger().info("Reading Locomo dataset")
        conversation_id = self._args.conversation
        with open("datasets\\locomo\\locomo10.json", "r", encoding="utf-8") as locomo_dataset:
            dataset = [
                DatasetSample(
                    sample_id=conversation_sample['sample_id'],
                    sample=DatasetSampleInstance(
                        # TODO: Store messages separately with their corresponding ids for retrieval evaluation
                        # See https://github.com/oyar99/HybridLongMemGPT/issues/3
                        context=[_parse_conversation(
                            conversation_sample['conversation'], self._args)],
                        qa=filter_questions([QuestionAnswer(
                            question_id=f'{conversation_sample["sample_id"]}-{i + 1}',
                            question=qa['question'],
                            answer=qa.get('answer') or qa.get(
                                'adversarial_answer'),
                            category=qa['category']
                        ) for i, qa in enumerate(conversation_sample['qa'])], self._args.questions, self._args.category)
                    )
                )
                for conversation_sample in json.load(locomo_dataset)
                if conversation_id is None or conversation_sample['_id'] == conversation_id
            ][:self._args.limit]

            self._dataset = dataset
            Logger().info(
                f"Locomo dataset read successfully. Total samples: {len(dataset)}")

            return dataset

    def build_system_prompt(self) -> dict[str, str]:
        """Builds a system prompt for QA tasks

        Returns:
            dict[str, str]: A dictionary where the key is a dataset sample instance and the value its system prompt
        """
        if not self._dataset:
            Logger().error("Dataset not read. Please read the dataset before building system prompts.")
            raise ValueError(
                "Dataset not read. Please read the dataset before building system prompts.")

        Logger().info("Building system prompts")

        system_prompt = {
            sample['sample_id']: QA_PROMPT.format(
                conversation=sample['sample']['context'][0])
            for sample in self._dataset
        }

        Logger().info("System prompts built successfully")

        return system_prompt
