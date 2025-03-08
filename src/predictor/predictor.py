"""Predictor module."""
import json
import os
from azure_open_ai.batch_deployment import queue_qa_batch_job
from logger.logger import Logger
from models.dataset import Dataset
from systems.bm25_rag.bm25_rag import BM25RAG


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

    # TODO: Move into dataset using composition instead of here in the predictor where we shouldn't have this type of logic
    if args.retrieval:
        Logger().info("Indexing documents")
        corpus = dataset.read_corpus()
        bm25_rag = BM25RAG()
        bm25_rag.index(corpus)

        questions = dataset.get_questions()

        output_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'output' + os.sep + 'retrieval_jobs')
        output_name = os.path.join(
            output_dir, f'retrieval_results_{Logger().get_run_id()}.jsonl')

        with open(output_name, 'a', encoding='utf-8') as f:
            for sample_id, question_set in questions.items():
                for question in question_set:
                    result = bm25_rag.retrieve(question['question'], k=100)

                    result_json = {
                        'custom_id': sample_id,
                        'question': question['question'],
                        'result': result
                    }
                    f.write(json.dumps(result_json) + '\n')
    else:
        queue_qa_batch_job(
            args.model, dataset, stop=args.noop)
