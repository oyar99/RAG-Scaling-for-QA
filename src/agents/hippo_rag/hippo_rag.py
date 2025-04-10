"""
Hippo RAG system for document retrieval

Take from https://github.com/OSU-NLP-Group/HippoRAG/tree/main

The following env variables need to be defined for the agent to function correctly:

OPENAI_API_KEY: <open ai key>
CUDA_VISIBLE_DEVICES: <gpu id>

"""

import os
# from hipporag import HippoRAG as HippoRAGModel
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset


class HippoRAG(Agent):
    """
    Hippo RAG system for document retrieval.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        super().__init__(args)

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

        embedding_model = 'Contriever'
        hipporag = None
        #hipporag = HippoRAGModel(
        #    save_dir=hipporag_dir,
        #    llm_model_name=self._args.model,
        #    embedding_model_name=embedding_model,
        #)

        hipporag.index(docs=[doc['content'] for doc in corpus])

        Logger().info("Successfully indexed documents")

        self._index = hipporag
        self._corpus = corpus
        self._qa_prompt = dataset.get_prompt('qa_rel')

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

    def batch_reason(self, _: list[str]) -> NoteBook:
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

        self._index.rag_qa(queries=questions)

        raise NotImplementedError(
            "HippoRAG agent is not fully implemented yet."
        )
