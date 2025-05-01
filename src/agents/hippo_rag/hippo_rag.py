"""
Hippo RAG system for document retrieval

Take from https://github.com/OSU-NLP-Group/HippoRAG/tree/main

The following env variables need to be defined for the agent to function correctly:

OPENAI_API_KEY: <open ai key>
CUDA_VISIBLE_DEVICES: <gpu id>

"""

import os
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult


class HippoRAG(Agent):
    """
    Hippo RAG system for document retrieval.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._reverse_doc_map = None
        super().__init__(args)

        self.standalone = True

    def index(self, dataset: Dataset) -> None:
        """
        Index the documents using HippoRAG agent

        Args:
            dataset (Dataset): The dataset to index
        """
        Logger().info("Indexing documents using HippoRAG agent")
        corpus = dataset.read_corpus()

        hipporag_dir = os.path.join(os.path.normpath(
            os.getcwd() + os.sep + os.pardir), 'temp' + os.sep + 'hipporag' + os.sep + dataset.name)

        os.makedirs(hipporag_dir, exist_ok=True)

        embedding_model = 'facebook/contriever'
        hipporag = None

        # pylint: disable-next=import-outside-toplevel
        from hipporag import HippoRAG as HippoRAGModel

        hipporag = HippoRAGModel(
            save_dir=hipporag_dir,
            llm_model_name=self._args.model,
            embedding_model_name=embedding_model,
            llm_base_url='http://localhost:8000/v1',
        )

        hipporag.index(docs=[doc['content'] for doc in corpus])

        Logger().info("Successfully indexed documents")

        self._index = hipporag
        self._corpus = corpus
        self._reverse_doc_map = {doc['content']: doc['doc_id'] for doc in corpus}

    def reason(self, _: str) -> NoteBook:
        """
        Perform reasoning on the question using the indexed documents.

        Args:
            question (str): The question to ask

        Returns:
            NoteBook: The notebook containing the retrieved documents
        """
        Logger().error(
            "HippoRAG agent does not support single question reasoning. Use multiprocessing_reason instead."
        )
        raise NotImplementedError(
            "HippoRAG agent does not support single question reasoning. Use multiprocessing_reason instead."
        )

    def batch_reason(self, _: list[QuestionAnswer]) -> list[NoteBook]:
        """
        Uses its question index to answer the questions.

        Raises:
            NotImplementedError: Batch reasoning is not implemented for the HippoRAG agent.
        """
        raise NotImplementedError(
            "Batch reasoning is not implemented for the HippoRAG agent.")

    def multiprocessing_reason(self, questions: list[str]) -> list[NoteBook]:
        """
        Perform reasoning on the questions using the indexed documents in parallel.

        Args:
            questions (list[str]): The questions to ask
        Returns:
            list[NoteBook]: The notebooks containing the retrieved documents
        """
        if self._index is None or not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")

        results = self._index.rag_qa(queries=questions)

        Logger().info("Successfully retrieved documents")

        notebooks = []

        for result in results[0]:
            retrieved_docs = [
                RetrievedResult(
                    doc_id=self._reverse_doc_map[doc],
                    content=doc,
                    score=float(score)
                ) for doc, score in zip(result.docs, result.doc_scores)
            ]

            notebook = NoteBook()
            notebook.update_sources(retrieved_docs)
            notebook.update_notes(result.answer[:1000])

            notebooks.append(notebook)

        return notebooks
