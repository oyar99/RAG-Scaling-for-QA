"""Evaluation baseline benchmark for the Locomo dataset."""
import argparse
from dotenv import load_dotenv

from evaluator.evaluator import evaluator
from logger.logger import Logger
from predictor.predictor import predictor


def parse_args():
    """Parses the command line arguments.

    Returns:
        dict: the parsed arguments
    """
    parser = argparse.ArgumentParser(
        prog='locomo-baseline-eval',
        description='Evaluate the Locomo benchmark using the baseline system'
    )

    parser.add_argument('-e', '--execution', choices=['eval', 'predict'], required=True,
                        help='mode of execution (required)')

    # Predict mode arguments
    parser.add_argument('-m', '--model', type=str,
                        help='model deployment identifier (required in predict mode)')
    parser.add_argument('-c', '--conversation', type=str,
                        help='conversation id to be predicted (optional)')
    parser.add_argument('-q', '--questions', type=int,
                        help='number of questions to be answered in each conversation during predict mode (optional)')
    parser.add_argument('-ct', '--category', type=int,
                        help='category to be predicted (optional)')

    parser.add_argument('-dt', '--dis-trunc', type=int, default=0,
                        help='disable truncation (optional)')

    parser.add_argument('-np', '--noop', type=int, default=0,
                        help='do not run actual prediction (optional)')

    # Evaluation mode arguments
    parser.add_argument('-ev', '--evaluation', type=str,
                        help='evaluation file path (required in evaluation mode)')

    parser.add_argument('-bt', '--bert-eval', type=int,
                        default=0, help='run bert evaluation (optional)')

    return parser.parse_args()


def main():
    """
    Entry point of the script
    """
    args = parse_args()
    load_dotenv()

    if args.execution == 'eval':
        evaluator(args)
    elif args.execution == 'predict':
        predictor(args)


if __name__ == "__main__":
    Logger().info(
        f"Starting program with execution id: {Logger().get_run_id()}")
    main()
