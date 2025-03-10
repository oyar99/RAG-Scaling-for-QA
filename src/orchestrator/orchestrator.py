"""Orchestrator module to run the predictor and evaluator."""

from typing import Type
from agents.bm25.bm25 import BM25
from agents.default.default import Default
from datasets.hotpot.hotpot import Hotpot
from datasets.locomo.locomo import Locomo
from datasets.musique.musique import MuSiQue
from datasets.twowikimultihopqa.two_wiki import TwoWiki
from evaluator.evaluator import evaluator
from logger.logger import Logger
from models.agent import Agent
from models.dataset import Dataset
from predictor.predictor import predictor

# pylint: disable-next=too-few-public-methods
class Orchestrator:
    """
    Orchestrator class to run the predictor and evaluator.
    It initializes the agent and dataset based on the provided arguments.
    """

    def __init__(self, args):
        self._config = args

        datasets: dict[str, Type[Dataset]] = {
            'locomo': Locomo,
            'hotpot': Hotpot,
            '2wiki': TwoWiki,
            'musique': MuSiQue,
        }

        if args.dataset not in datasets:
            Logger().error(f"Dataset {args.dataset} not supported")
            raise ValueError(f"Dataset {args.dataset} not supported")

        agents: dict[str, Type[Agent]] = {
            'default': Default,
            'bm25': BM25,
        }

        if args.agent not in agents:
            Logger().error(f"Agent {args.agent} not supported")
            raise ValueError(f"Agent {args.agent} not supported")

        self.agent = agents[args.agent](args)
        self.dataset = datasets[args.dataset](args)

    def run(self):
        """
        Generates predictions for the given dataset using the specified agent.

        Raises:
            ValueError: if the execution mode is not supported
        """
        if self._config.execution == 'predict':
            Logger().info("Running predictor")
            predictor(self._config, self.dataset, self.agent)
        elif self._config.execution == 'eval':
            Logger().info("Running predictor")
            evaluator(self._config, self.dataset)
        else:
            Logger().error(
                f"Execution mode {self._config.execution} not supported")
            raise ValueError(
                f"Execution mode {self._config.execution} not supported")
