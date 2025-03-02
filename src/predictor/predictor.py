"""Predictor module."""
from azure_open_ai.batch_deployment import queue_qa_batch_job
from logger.logger import Logger
from models.dataset import Dataset


def predictor(args, dataset: Dataset) -> None:
    """
    Generates predictions for the given conversation.

    Args:
        args (Namespace): the arguments passed to the script
        dataset (Dataset): the dataset to be processed
        
    Raises:
        ValueError: if the model deployment identifier is not provided
    """
    if args.model is None:
        Logger().error(
            """Model deployment identifier not provided. \
Please provide the model deployment identifier using the -m flag.""")
        raise ValueError("Model deployment identifier not provided")

    _ = dataset.read()

    queue_qa_batch_job(
        args.model, dataset, stop=args.noop)
