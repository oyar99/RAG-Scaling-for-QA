"""Evaluator module."""

import os
import json
from typing import Optional
from evaluator.exact_match_evaluator import eval_exact_match
from evaluator.f1_evaluator import eval_f1_score
from evaluator.bert_evaluator import eval_bert_score
from evaluator.retrieval_evaluator import eval_retrieval_recall
from logger.logger import Logger
from models.dataset import Dataset
from models.document import Document


def evaluator(args, dataset: Dataset) -> None:
    """
    Orchestrates the evaluation of the model's performance on the dataset.
    It evaluates the model's performance based on the provided arguments and dataset.

    Args:
        args (Namespace): the arguments passed to the script
        dataset (Dataset): the dataset to be processed

    Raises:
        ValueError: if the evaluation file is not provided
    """

    # Need to override these parameters to ensure that question ids which
    # depend on the original index can be correctly mapped for LoCoMo dataset
    # TODO: Figure out better solution. See https://github.com/oyar99/HybridLongMemGPT/issues/4
    args.questions = None
    args.category = None
    _ = dataset.read()

    Logger().info("Running evaluation")

    if args.evaluation is None:
        Logger().error("Evaluation file not provided. Please provide the evaluation file path using the -ev flag.")
        raise ValueError("Evaluation file not provided")

    with open(args.evaluation, "r", encoding="utf-8") as evaluation_file:
        evaluation = [json.loads(line) for line in evaluation_file]

        def extract_doc_pair(eval_item) -> Optional[tuple[list[Document], list[Document]]]:
            Logger().debug(
                f"Extracting docs pair for evaluation item: {eval_item['custom_id']}")
            question = dataset.get_question(eval_item['custom_id'])

            if question is None:
                Logger().warn(
                    f"Sample id {eval_item['custom_id']} not found in the dataset. Skipping evaluation ...")
                return None

            Logger().debug(f"Question found: {question['question']}")

            expected_docs = dataset.get_supporting_docs(eval_item['custom_id'])
            actual_docs = [Document(
                doc_id=result['doc_id'],
                content=result['content']
            ) for result in eval_item['result']]

            doc_pairs = (
                expected_docs,
                actual_docs,
            )

            Logger().debug(
                f"Retrieval extracted: (question, truth, predicted): {question['question']}{doc_pairs}")

            return doc_pairs

        def extract_qa_pair(eval_item) -> Optional[tuple[list[str], str]]:
            Logger().debug(
                f"Extracting QA pair for evaluation item: {eval_item['custom_id']}")
            question = dataset.get_question(eval_item['custom_id'])

            if question is None:
                Logger().warn(
                    f"Sample id {eval_item['custom_id']} not found in the dataset. Skipping evaluation ...")
                return None

            Logger().debug(f"Question found: {question['question']}")

            qa_pair = (str(question['answer']),
                       str(eval_item['response']['body']['choices'][0]['message']['content']))

            Logger().debug(
                f"QA extracted: (question, truth, predicted): {question['question']}{qa_pair}")
            return qa_pair

        if args.retrieval:
            doc_pairs = [pair for pair in (extract_doc_pair(
                eval_item) for eval_item in evaluation) if pair is not None]
            evaluate_retrieval(doc_pairs)
        else:
            qa_pairs = [pair for pair in (extract_qa_pair(
                eval_item) for eval_item in evaluation) if pair is not None]
            evaluate(qa_pairs, args)


def evaluate_retrieval(doc_pairs: list[tuple[list[Document], list[Document]]]) -> None:
    """
    Evaluates retrieval performance based on the provided document pairs.
    Evaluates the recall score across various Ks.

    Args:
        doc_pairs (list[tuple[list[Document], list[Document]]]): the ground truth documents and the model's documents
    """
    Logger().info("Evaluating retrieval score")

    recall_at_k = eval_retrieval_recall(doc_pairs)

    for k, recall in recall_at_k.items():
        Logger().info(f"Recall at {k}: {recall}")

    output_dir = os.path.join(os.path.normpath(
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'retrieval')
    output_name = os.path.join(output_dir, f'eval-{Logger().get_run_id()}.out')

    with open(output_name, "w", encoding="utf-8") as output_file:
        for k, recall in recall_at_k.items():
            output_file.write(f"Recall at {k}: {recall}\n")


def evaluate(qa_pairs: list[tuple[list[str], str]], args) -> None:
    """
    Evaluates question answering performance based on the provided question-answer pairs.
    Evaluates the exact match score, F1 score, precision, recall, and BERT score (if applicable).

    Args:
        qa_pairs (list[tuple[list[str], str]]): the ground truth answers and the model's answers

        args (Namespace): the arguments passed to the script
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
        os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'qa')
    output_name = os.path.join(output_dir, f'eval-{Logger().get_run_id()}.out')

    with open(output_name, "w", encoding="utf-8") as output_file:
        output_file.write(f"Exact match score: {em}\n")
        output_file.write(f"F1 score: {f1}\n")
        output_file.write(f"Precision: {precision}\n")
        output_file.write(f"Recall: {recall}\n")
        if bert_score:
            output_file.write(f"BERT score: {bert_score}\n")
