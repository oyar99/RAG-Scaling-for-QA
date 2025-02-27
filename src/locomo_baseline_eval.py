"""Evaluation baseline benchmark for the Locomo dataset."""
import argparse
import json
import re
from collections import defaultdict
from dotenv import load_dotenv

from azure_open_ai.batch_deployment import queue_qa_batch_job
from logger.logger import Logger
from models.question import Question

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
conversation between two users followed by a question. You need to provide a concise answer using exact \
words from the conversations when possible. \

The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation:

Below is the conversation.

{conversation}
'''


def parse_args():
    """Parses the command line arguments.

    Returns:
        dict: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='locomo-baseline-eval',
        description='Evaluate the Locomo benchmark using the baseline system'
    )

    parser.add_argument('-m', '--model', type=str,
                        required=True, help='model deployment identifier')
    parser.add_argument('-c', '--conversation', type=str,
                        help='conversation id to be evaluated (optional)')

    parser.add_argument('-q', '--questions', type=int,
                        help='number of questions to be answered in each conversation (optional)')

    parser.add_argument('-ct', '--category', type=int,
                        help='category to be evaluated (optional)')

    return parser.parse_args()


def read_dataset(conversation_id: str = None) -> list[dict]:
    """Reads the Locomo dataset from the datasets directory.

    Args:
        conversation_id (str, optional): _description_. Defaults to None.

    Returns:
        list[dict]: the dataset as a list of dictionaries
    """
    Logger().info("Reading Locomo dataset")
    with open("datasets\\locomo\\locomo10.json", "r", encoding="utf-8") as locomo_dataset:
        return [
            {
                **conversation_sample,
                'qa': [{**qa, 'id': f'{conversation_sample["sample_id"]}-{i + 1}'}
                       for i, qa in enumerate(conversation_sample['qa'])]
            }
            for conversation_sample in (json.load(locomo_dataset) if not conversation_id else [
                sample for sample in json.load(locomo_dataset) if sample['sample_id'] == conversation_id])
        ]


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

    if limit is not None:
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


def main():
    """
    Entry point of the script. Parses arguments, reads the dataset, builds system prompts and 
    questions, and queues the batch job for evaluation.
    """
    args = parse_args()
    load_dotenv()

    dataset = read_dataset(args.conversation)

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


if __name__ == "__main__":
    Logger().info(
        f"Starting Locomo baseline evaluation. RunId: {Logger().get_run_id()}")
    main()
