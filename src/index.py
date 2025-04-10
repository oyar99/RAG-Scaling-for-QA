"""Evaluating agent-based architectures for retrieval and answer generation tasks."""
import argparse
from typing import Any
from dotenv import load_dotenv

from logger.logger import Logger
from orchestrator.orchestrator import Orchestrator


def parse_args() -> dict[str, Any]:
    """
    Parses the command line arguments.

    Returns:
        dict[str, Any]: parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='agent-eval-mem',
        description='Evaluate various agent-based architectures for retrieval and answer generation tasks'
    )

    parser.add_argument('-e', '--execution', choices=['eval', 'predict'], required=True,
                        help='mode of execution (required)')

    # Dataset processing arguments
    parser.add_argument('-d', '--dataset', choices=['locomo', 'hotpot', '2wiki', 'musique'], required=True,
                        help='dataset to be processed (required)')
    parser.add_argument('-c', '--conversation', type=str,
                        help='conversation id to be extracted from the dataset - (optional)')
    parser.add_argument('-q', '--questions', type=int,
                        help='number of questions to be processed in each dataset sample (optional)')
    parser.add_argument('-ct', '--category', type=int,
                        help='category to be extracted from the dataset (optional)')
    parser.add_argument('-l', '--limit', type=int,
                        help='limit the number of samples to process. \
Ignored if conversation id is provided (optional)')

    # Predict mode arguments
    parser.add_argument('-m', '--model', choices=['gpt-4o-mini', 'o3-mini'],
                        help='model deployment identifier (required in predict mode)')

    parser.add_argument('-a', '--agent', choices=['default', 'oracle', 'bm25', 'dense', 'colbertv2', 'hippo'],
                        default='default', help='agent to be used (required in predict mode)')

    parser.add_argument('-np', '--noop', action='store_true',
                        help='do not run actual prediction (optional)')

    # Evaluation mode arguments
    parser.add_argument('-ev', '--evaluation', type=str,
                        help='evaluation file path (required in evaluation mode)')

    parser.add_argument('-bt', '--bert-eval', action='store_true',
                        help='run bert evaluation (optional)')

    parser.add_argument('-j', '--judge-eval', action='store_true',
                        help='run judge evaluation (optional). Other evaluations will be skipped in this mode')

    parser.add_argument('-jp', '--judge-eval-path', type=str,
                        help='path to the judge evaluation file (optional). \
If not provided, an evaluation file is generated')

    parser.add_argument('-r', '--retrieval', action='store_true',
                        help='run retrieval evaluation (optional)')

    parser.add_argument('-eb', '--eval-batch', action='store_true',
                        help='run batch evaluation (optional)')

    return parser.parse_args()


def main():
    """
    Entry point to evaluate agent-based architectures for retrieval and answer generation tasks.
    """
    args = parse_args()

    Orchestrator(args).run()


if __name__ == "__main__":
    load_dotenv()
    Logger().info(
        f"Starting program with execution id: {Logger().get_run_id()}")
    main()
