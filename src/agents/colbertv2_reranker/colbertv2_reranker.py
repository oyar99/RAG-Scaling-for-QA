"""ColbertV2 with LLM reranker RAG system for document retrieval and question answering."""
import os
import json
from colbert.infra import Run, RunConfig
from colbert import Searcher
from agents.colbertv2.colbertv2 import ColbertV2
from azure_open_ai.batch import queue_batch_job, wait_for_batch_job_and_save_result
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult
from utils.model_utils import supports_temperature_param
from utils.token_utils import get_max_output_tokens


class ColbertV2Reranker(Agent):
    """
    ColbertV2 RAG system for document retrieval and question answering using late interaction.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        self._colbertv2 = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval.
        """
        self._colbertv2 = ColbertV2(self._args)
        self._colbertv2.index(dataset)

        self._index = dataset.name or 'index'
        # pylint: disable-next=protected-access
        self._corpus = self._colbertv2._corpus
        self._qa_prompt = dataset.get_prompt('qa_rel')

    def reason(self, _: str) -> NoteBook:
        """
        Reason over the indexed dataset to answer the question.
        """
        Logger().error(
            "ColBERTV2 agent does not support single question reasoning. Use multiprocessing_reason instead."
        )
        raise NotImplementedError(
            "ColBERTV2 agent does not support single question reasoning. Use multiprocessing_reason instead."
        )

    def batch_reason(self, _: list[str]) -> NoteBook:
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the colbertv2 agent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the colbertv2 agent.")

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Multiprocessing reason over the indexed dataset to answer the questions.

        Args:
            questions (list[str]): List of questions to answer.
        """
        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            searcher = Searcher(index=self._index, collection=[
                                doc['content'] for doc in self._corpus])

        Logger().info("Searching for answers to questions")

        results = searcher.search_all(queries=dict(enumerate(questions)), k=10)

        grouped_results = {}

        Logger().info("Processing results")

        for q_id, doc_id, _, score in results.flat_ranking:
            if q_id not in grouped_results:
                grouped_results[q_id] = []
            grouped_results[q_id].append((doc_id, score))

        os.makedirs(os.path.join(colbert_dir, 'tmp'), exist_ok=True)
        rerank_results_path = os.path.join(
            colbert_dir, 'tmp/rerank_results.jsonl')

        batch = queue_batch_job([
            {
                "custom_id": q_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._args.model,
                    "messages": [
                        {"role": "system",

                         "content": RERANKER_PROMPT.format(
                             documents=str([self._corpus[d_id]['content'] for d_id, _ in docs]))},
                        {"role": "user", "content": questions[q_id]}
                    ],
                    "temperature": (default_job_args['temperature']
                                    if supports_temperature_param(self._args.model) else None),
                    "frequency_penalty": default_job_args['frequency_penalty'],
                    "presence_penalty": default_job_args['presence_penalty'],
                    "max_completion_tokens": int(get_max_output_tokens(self._args.model))
                },
            }
            for q_id, docs in grouped_results.items()
        ])

        Logger().info("Waiting for batch job to finish")

        wait_for_batch_job_and_save_result(batch, rerank_results_path)

        notebooks_dict = {}

        with open(rerank_results_path, 'r', encoding='utf-8') as file:
            for line in file:
                result = json.loads(line)
                q_id = result['custom_id']

                new_ranking = None

                try:
                    # Extract only the last line of the response content
                    last_line = next(
                        (line for line in
                         reversed(
                             result['response']['body']['choices'][0]['message']['content'].strip().split('\n'))
                         if line.startswith('[')), None)

                    candidate_ranking = json.loads(last_line) if last_line else None

                    # Validate candidate_ranking
                    if (isinstance(candidate_ranking, list) and
                            len(candidate_ranking) == 10 and
                            sorted(candidate_ranking) == list(range(1, 11))):
                        new_ranking = candidate_ranking
                        if candidate_ranking != list(range(1, 11)):
                            Logger().info(
                                f"New ranking for question ID {q_id}: {new_ranking}")
                    else:
                        Logger().warn(
                            f"Invalid ranking format for question ID {q_id}: {candidate_ranking}")
                        Logger().info("Use the original ranking")
                        new_ranking = list(range(1, 11))
                except json.JSONDecodeError:
                    Logger().warn(
                        f"Failed to decode JSON response for question ID {q_id}: \
{result['response']['body']['choices'][0]['message']['content']}")
                    Logger().info("Use the original ranking")
                    new_ranking = list(range(1, 11))

                ranked_docs = [grouped_results[q_id][rank-1]
                               for rank in new_ranking]

                retrieved_results = [
                    RetrievedResult(
                        doc_id=self._corpus[doc_id]['doc_id'],
                        content=self._corpus[doc_id]['content'],
                        score=score
                    )
                    for doc_id, score in ranked_docs[:5]
                ]

                notebook = NoteBook()
                notebook.update_sources(retrieved_results)

                notes = self._qa_prompt.format(
                    context='\n'.join(
                        doc['content']
                        for doc in retrieved_results)
                )

                notebook.update_notes(notes)
                notebooks_dict[int(q_id)] = notebook

        return [notebooks_dict[key] for key in sorted(notebooks_dict.keys())]


default_job_args = {
    'temperature': 0.0,
    'max_completion_tokens': 100,
    'frequency_penalty': 0.0,
    'presence_penalty': 0.0
}


RERANKER_PROMPT = '''You are tasked with re-ranking 10 documents based on their relevance to a given question.\
The documents are initially ranked using the ColBERTV2 model.

Your response should be a valid JSON array of numbers from 1 to 10, where:

- Each number appears exactly once.\n
- Each number corresponds to the position of a document in the original ranking.\n
- The position of each number in the array represents the new rank of the document.\n
- The number at the first position is the most relevant document, and the number at the last position is the least relevant.\n

For example, given the following documents:

["France is a country in Europe.", \
"Paris is the capital of France.", \
"Berlin is the capital of Germany.", \
"London is the capital of the UK.", \
"Rome is the capital of Italy.",  \
"Spain is a country in Europe.", \
"Paris is one of the largest capitals in Europe.", \
"The Eiffel Tower is in Paris.", \
"France is known for its wine.", \
"Germany is known for its beer."],

and the question "What is the capital of France?", your response should be:

[2, 7, 1, 8, 9, 3, 4, 5, 6, 10]

Let us think step by step.

Finally, the last line of your response should be a valid JSON array of numbers.

Below are the documents for re-ranking.

{documents}
'''
