"""Evaluator module."""

import os
import json
import re
from datasets.locomo.locomo import Locomo
from evaluator.exact_match_evaluator import eval_exact_match
from evaluator.f1_evaluator import eval_f1_score
from evaluator.bert_evaluator import eval_bert_score
from logger.logger import Logger


def evaluator(args) -> None:
    """
    Evaluates the exact match between the ground truth answers and the model's answers.

    Args:
        args (Namespace): the arguments passed to the script
    """

    locomo = Locomo(args)
    dataset = locomo.read()

    Logger().info(
        f"Locomo dataset read successfully. Total samples: {len(dataset)}")

    Logger().info("Running evaluation")

    if args.evaluation is None:
        Logger().error("Evaluation file not provided. Please provide the evaluation file path using the -ev flag.")
        return

    with open(args.evaluation, "r", encoding="utf-8") as evaluation_file:
        evaluation = [json.loads(line) for line in evaluation_file]

        dataset_map = {sample['sample_id']: sample['sample'] for sample in dataset}

        def extract_qa_pair(eval_item) -> tuple[str, str]:
            Logger().debug(
                f"Extracting QA pair for evaluation item: {eval_item['custom_id']}")
            match = re.match(r'^(conv-.\d+)-(\d+)$',
                             eval_item['custom_id'])
            sample_id = match.group(1)
            message_id = match.group(2)
            conversation_obj = dataset_map[sample_id]['qa'][int(
                message_id)-1]
            qa_pair = (str(conversation_obj['answer']),
                       str(eval_item['response']['body']['choices'][0]['message']['content']))

            Logger().debug(
                f"QA extracted: (question, truth, predicted): {conversation_obj['question']}{qa_pair}")
            return qa_pair

        evaluate([extract_qa_pair(eval_item)
                  for eval_item in evaluation], args)


def evaluate(qa_pairs: list[tuple[str, str]], args) -> None:
    """
    Evaluates the exact match between the ground truth answers and the model's answers.

    Args:
        qa_pairs (list[tuple[str, str]]): the ground truth answers and the model's answers

        args (Namespace): the arguments passed to the script

    Returns:
        float: the evaluation score
    """

    Logger().info("Evaluating exact match score")

    em = eval_exact_match(qa_pairs)
    f1, precision, recall = eval_f1_score(qa_pairs)
    bert_score = None

    if args.bert_eval:
        bert_score = eval_bert_score(qa_pairs)
        Logger().info(f"BERT score: {bert_score}")

    Logger().info(f"Exact match score: {em}")
    Logger().info(f"F1 score: {f1}")
    Logger().info(f"Precision: {precision}")
    Logger().info(f"Recall: {recall}")

    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output')
    output_name = os.path.join(output_dir, f'eval-{Logger().get_run_id()}.out')

    with open(output_name, "w", encoding="utf-8") as output_file:
        output_file.write(f"Exact match score: {em}")
        output_file.write(f"F1 score: {f1}")
        output_file.write(f"Precision: {precision}")
        output_file.write(f"Recall: {recall}")
        if bert_score:
            output_file.write(f"BERT score: {bert_score}")
