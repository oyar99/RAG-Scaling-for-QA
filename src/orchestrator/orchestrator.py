"""Orchestrator module to run the predictor and evaluator."""

from typing import Type
from datasets.hotpot.hotpot import Hotpot
from datasets.locomo.locomo import Locomo
from evaluator.evaluator import evaluator
from logger.logger import Logger
from models.dataset import Dataset
from predictor.predictor import predictor

# pylint: disable-next=too-few-public-methods
class Orchestrator:
    """An orchestrator class to run the predictor and evaluator.
    """

    def __init__(self, args):
        self._config = args

        datasets: dict[str, Type[Dataset]] = {
            'locomo': Locomo,
            'hotpot': Hotpot
        }

        if args.dataset not in datasets:
            Logger().error(f"Dataset {args.dataset} not supported")
            raise ValueError(f"Dataset {args.dataset} not supported")

        self.dataset = datasets[args.dataset](args)

    def run(self):
        """
        Generates predictions for the given conversation.
        """
        if self._config.execution == 'predict':
            Logger().info("Running predictor")
            predictor(self._config, self.dataset)
        elif self._config.execution == 'eval':
            Logger().info("Running predictor")
            evaluator(self._config, self.dataset)
        else:
            Logger().error(
                f"Execution mode {self._config.execution} not supported")
            raise ValueError(
                f"Execution mode {self._config.execution} not supported")
