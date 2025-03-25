"""Dense RAG system for document retrieval using dense embeddings."""

from sentence_transformers import SentenceTransformer, util
import torch
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.retrieved_result import RetrievedResult


class Dense(Agent):
    """
    Dense RAG system for document retrieval using dense embeddings.
    """

    def __init__(self, args):
        self._index = None
        self._corpus = None
        self._qa_prompt = None
        self._sentence_transformer = None
        super().__init__(args)

    def index(self, dataset: Dataset) -> None:
        """
        Index the documents using dense embeddings.

        Args:
            dataset (Dataset): The dataset to index
        """
        Logger().info("Indexing documents using Dense agent")
        corpus = dataset.read_corpus()

        sentence_transformer = SentenceTransformer(
            'sentence-transformers/msmarco-bert-base-dot-v5',
            model_kwargs={'torch_dtype': torch.float16},
        )

        devices = [f"cuda:{i}"
                   for i in range(min(torch.cuda.device_count(), 2))] if torch.cuda.is_available() else ['cpu']

        Logger().info(f"Using devices: {devices}")

        pool = sentence_transformer.start_multi_process_pool(
            target_devices=devices)

        Logger().info(
            f"Max sequence length: {sentence_transformer.max_seq_length}")

        corpus_embeddings = sentence_transformer.encode_multi_process(
            [doc['content']
             for doc in corpus],
            show_progress_bar=True,
            pool=pool,
        )

        sentence_transformer.stop_multi_process_pool(pool=pool)

        Logger().info("Successfully indexed documents")
        self._index = corpus_embeddings
        self._corpus = corpus
        self._qa_prompt = dataset.QA_PROMPT
        self._sentence_transformer = sentence_transformer

    def reason(self, _: str) -> NoteBook:
        """
        Perform reasoning on the question using the indexed documents.

        Args:
            question (str): The question to ask

        Returns:
            NoteBook: The notebook containing the retrieved documents
        """
        Logger().error(
            "Dense agent does not support single question reasoning. Use multiprocessing_reason instead."
        )
        raise NotImplementedError(
            "Dense agent does not support single question reasoning. Use multiprocessing_reason instead."
        )

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

        devices = [f"cuda:{i}"
                   for i in range(min(torch.cuda.device_count(), 2))] if torch.cuda.is_available() else ['cpu']

        Logger().info(f"Using devices: {devices}")

        pool = self._sentence_transformer.start_multi_process_pool(
            target_devices=devices)

        query_embeddings = self._sentence_transformer.encode_multi_process(
            questions,
            show_progress_bar=True,
            pool=pool,
        )

        self._sentence_transformer.stop_multi_process_pool(pool=pool)

        Logger().info("Successfully computed query embeddings")

        notebook = NoteBook()
        k = 5

        scores_matrix = util.dot_score(query_embeddings, self._index)

        Logger().info("Successfully computed scores matrix")

        notebooks = []
        for scores in scores_matrix:
            # Get the top k indices for each query
            top_k_indices = torch.topk(scores, k, largest=True).indices

            retrieved_docs = [
                RetrievedResult(
                    doc_id=self._corpus[idx]['doc_id'],
                    content=self._corpus[idx]['content'],
                    score=scores[idx].item()
                ) for idx in top_k_indices
            ]

            # Create a notebook for each query
            notebook = NoteBook()
            notebook.update_sources(retrieved_docs)

            notes = self._qa_prompt.format(
                context='\n'.join(
                    doc['content']
                    for doc in retrieved_docs)
            )

            notebook.update_notes(notes)
            notebooks.append(notebook)

        return notebooks
