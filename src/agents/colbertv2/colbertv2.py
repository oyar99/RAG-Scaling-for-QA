"""ColbertV2 RAG system for document retrieval and question answering."""
import os
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult


class ColbertV2(Agent):
    """
    ColbertV2 RAG system for document retrieval and question answering using late interaction.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the dataset for retrieval.
        """
        Logger().info("Indexing documents using ColbertV2")
        corpus = dataset.read_corpus()

        colbert_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'colbert')

        os.makedirs(colbert_dir, exist_ok=True)

        with Run().context(RunConfig(nranks=2, experiment=os.path.join(colbert_dir, 'colbertv2.0'))):
            config = ColBERTConfig(
                nbits=2,
            )
            self._index = Indexer('colbert-ir/colbertv2.0', config=config)
            self._index.index(
                name=dataset.name or 'index',
                collection=[doc['content'] for doc in corpus],
                overwrite='reuse'
            )

        self._index = dataset.name or 'index'
        self._corpus = corpus
        self._qa_prompt = dataset.get_prompt('qa_rel')
        Logger().info("Successfully indexed documents")

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

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:
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

        results = searcher.search_all(queries=dict(enumerate(questions)), k=5)

        notebooks = []

        grouped_results = {}

        Logger().info("Processing results")

        for q_id, doc_id, _, score in results.flat_ranking:
            if q_id not in grouped_results:
                grouped_results[q_id] = []
            grouped_results[q_id].append((doc_id, score))

        for result in grouped_results.values():
            retrieved_docs = [
                RetrievedResult(
                    doc_id=self._corpus[doc_id]['doc_id'],
                    content=self._corpus[doc_id]['content'],
                    score=score
                ) for doc_id, score in result
            ]

            # pylint: disable=duplicate-code
            notebook = NoteBook()
            notebook.update_sources(retrieved_docs)

            notes = self._qa_prompt.format(
                context='\n'.join(
                    doc['content']
                    for doc in retrieved_docs)
            )

            notebook.update_notes(notes)
            notebooks.append(notebook)
            # pylint: enable=duplicate-code

        return notebooks
