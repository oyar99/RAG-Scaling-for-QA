"""Predictor module."""
from azure_open_ai.batch_deployment import queue_qa_batch_job
from datasets.locomo.locomo import Locomo
from logger.logger import Logger


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

    locomo = Locomo(args)
    _ = locomo.read()

    queue_qa_batch_job(
        args.model, locomo, stop=args.noop)
