"""Predictor module."""
from collections import defaultdict
import re

from azure_open_ai.batch_deployment import queue_qa_batch_job
from datasets.locomo.read_locomo import read_locomo
from logger.logger import Logger
from models.question import Question
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


def predictor(args):
    """
    Generates predictions for the given conversation.

    Args:
        args (Namespace): the arguments passed to the script
    """
    if args.model is None:
        Logger().error(
            """Model deployment identifier not provided. \
Please provide the model deployment identifier using the -m flag.""")
        return

    dataset = read_locomo(args.conversation)

    Logger().info(
        f"Locomo dataset read successfully. Total samples: {len(dataset)}")

    Logger().info("Building system prompts")

    system_prompt = {
        conversation_sample['sample_id']: build_system_prompt(
            conversation_sample['conversation'], args)
        for conversation_sample in dataset
    }

    Logger().info("Building questions")

    questions = [
        Question(
            question_id=qa['id'],
            question=qa['question'],
            conversation_id=conversation_sample['sample_id'],
            category=qa['category']
        )
        for conversation_sample in dataset
        for qa in (filter_questions(conversation_sample['qa'], args.questions, args.category))
    ]

    Logger().info(f"Questions length: {len(questions)}")

    queue_qa_batch_job(
        args.model, system_prompt=system_prompt, questions=questions, stop=args.noop)


def filter_questions(questions: list, limit: int = None, category: int = None) -> list:
    """Filters the questions based on the category and limit.

    Args:
        questions (list): the list of questions
        limit (int, optional): the limit of questions to be returned
        category (int): the category to be returned. All if not specified

    Returns:
        list: the filtered list of questions
    """
    filtered_questions = [
        question for question in questions
        if category is None or question['category'] == category
    ]

    if limit is not None and limit < len(filtered_questions):
        filtered_questions = filtered_questions[:limit]

    return filtered_questions


def build_system_prompt(conversation, args) -> str:
    """Builds the system prompt for the model.

    Args:
        conversation (dict): the conversation object between two users

        args (Namespace): the arguments passed to the script

    Returns:
        str: the system prompt
    """
    return QA_PROMPT.format(conversation=parse_conversation(conversation, args))


def parse_conversation(conversation, args) -> str:
    """Parses the conversation object in a human-readable format.

    Args:
        conversation (dict): the conversation object between two users

        args (Namespace): the arguments passed to the script

    Returns:
        str: a human readable string representation of the conversation
    """
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

    return truncate_conversation(conversation, args.model)


def truncate_conversation(conversation: str, model: str) -> str:
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
        conversation = conversation[-15000:]

    return conversation
