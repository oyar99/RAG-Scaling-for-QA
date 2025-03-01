"""Locomo dataset reader."""
import json
from logger.logger import Logger


def read_locomo(conversation_id: str = None) -> list[dict]:
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
        