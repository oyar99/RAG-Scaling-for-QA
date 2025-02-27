import argparse
import json

from AzureOpenAI.batch_deployment import queue_qa_batch_job
from logger.logger import Logger
from models.question import Question
from dotenv import load_dotenv
import re
from collections import defaultdict

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
question and you need to provide a concise answer with no explanations based only on the provided conversation between two people. \

The conversation takes place over multiple days and the date of each conversation is written at the beginning of the conversation:

Below is the conversation between the two users.

{conversation}
'''


def parse_args():
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

    return parser.parse_args()


def read_dataset(conversation_id: str = None):
    Logger().info("Reading Locomo dataset")
    with open("datasets\locomo\locomo10.json", "r") as locomo_dataset:
        return [
            {
                **conversation_sample,
                'qa': [{**qa, 'id': f'{conversation_sample["sample_id"]}-{i + 1}'} for i, qa in enumerate(conversation_sample['qa'])]
            }
            for conversation_sample in (json.load(locomo_dataset) if not conversation_id else [
                sample for sample in json.load(locomo_dataset) if sample['sample_id'] == conversation_id])
        ]


def parse_conversation(conversation):
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


def build_system_prompt(conversation):
    return QA_PROMPT.format(conversation=parse_conversation(conversation))


def main():
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
            conversation_id=conversation_sample['sample_id']
        )
        for conversation_sample in dataset
        for qa in (conversation_sample['qa'] if args.questions is None else conversation_sample['qa'][:args.questions])
    ]

    Logger().info(f"Questions length: {len(questions)}")

    queue_qa_batch_job(
        args.model, system_prompt=system_prompt, questions=questions)


if __name__ == "__main__":
    Logger().info(
        f"Starting Locomo baseline evaluation. RunId: {Logger().get_run_id()}")
    main()
