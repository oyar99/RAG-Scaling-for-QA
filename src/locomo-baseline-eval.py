import argparse
import json

from AzureOpenAI.batch_deployment import queue_qa_batch_job
from models.question import Question
from dotenv import load_dotenv

QA_PROMPT = '''You are a helpful Question Answering assistant. You will be presented with a \
question and you need to provide a short concise answer based only on the provided conversation between two people. \

The conversation takes place over multiple days called sessions and is formatted as a json file with the following keys:
    
- speaker_a: The name of the first speaker in the conversation.
- speaker_b: The name of the second speaker in the conversation.
- session_x_date_time: The date and time of the session x in the conversation.
- session_x: An array of messages.
    
Each message is formatted as a dictionary with the following keys:
    
- speaker: The name of the speaker.
- dia_id: The dialogue id of the message.
- text: The text of the message.
    
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

    return parser.parse_args()


def read_dataset():
    with open("datasets\locomo\locomo10.json", "r") as locomo_dataset:
        return [
            {
                **conversation_sample,
                'qa': [{**qa, 'id': f'{conversation_sample["sample_id"]}-{i + 1}'} for i, qa in enumerate(conversation_sample['qa'])]
            }
            for conversation_sample in json.load(locomo_dataset)
        ]


def build_system_prompt(conversation):
    return QA_PROMPT.format(conversation=conversation)


def main():
    args = parse_args()
    load_dotenv()
    dataset = read_dataset()
    system_prompt = {
        conversation_sample['sample_id']: build_system_prompt(conversation_sample['conversation'])
        for conversation_sample in dataset
    }
    questions = [
        Question(
            question_id=qa['id'],
            question=qa['question'],
            conversation_id=conversation_sample['sample_id']
        )
        for conversation_sample in dataset
        for qa in conversation_sample['qa']
    ]
    
    queue_qa_batch_job(args.model, system_prompt=system_prompt, questions=questions)


if __name__ == "__main__":
    main()
