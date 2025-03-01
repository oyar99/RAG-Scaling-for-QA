"""Predictor module."""
from collections import defaultdict
import re

from azure_open_ai.batch_deployment import queue_qa_batch_job
from datasets.locomo.read_locomo import read_locomo
from logger.logger import Logger
from models.question import Question

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
conversation between two users followed by a question. You need to provide a concise answer using exact \
words from the conversations when possible. Your answer should not contain any explanations or repeated sentences \
from the question itself. \

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
            conversation_sample['conversation'])
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
        args.model, system_prompt=system_prompt, questions=questions)


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


def build_system_prompt(conversation) -> str:
    """Builds the system prompt for the model.

    Args:
        conversation (dict): the conversation object between two users

    Returns:
        str: the system prompt
    """
    return QA_PROMPT.format(conversation=parse_conversation(conversation))


def parse_conversation(conversation) -> str:
    """Parses the conversation object in a human-readable format.

    Args:
        conversation (dict): the conversation object between two users

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

    return "\n".join(
        [
            f"DATE: {conversation[f'{session}_date_time']}\nCONVERSATION:\n" + "\n".join(
                [f"{message['speaker']} said: {message['text']}" for message in messages]
            )
            for session, messages in grouped_messages.items()
        ]
    )
