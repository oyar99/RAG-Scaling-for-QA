"""Default system for document retrieval that chooses docs directly from the dataset."""

from itertools import repeat
from multiprocessing import Pool, cpu_count
from typing import Union

import tiktoken
from logger.logger import Logger
from models.agent import Agent, NoteBook
from models.dataset import Dataset
from models.document import Document
from models.question_answer import QuestionAnswer
from models.retrieved_result import RetrievedResult
from utils.token_utils import get_encoding


class Default(Agent):
    """Default System"""

    def __init__(self, args):
        self._corpus = None
        self._index = None
        self._qa_prompt = None
        super().__init__(args)

        # Support batch reasoning
        self.support_batch = True

    def index(self, dataset: Dataset) -> None:
        """Index the documents

        Args:
            dataset (Dataset): The dataset to index
        """
        corpus = dataset.read_corpus()
        self._corpus = corpus
        self._index = corpus
        self._qa_prompt = dataset.get_prompt('qa_all')

    def process_batch(
        self,
        questions: list[QuestionAnswer],
        grouped_docs: dict[Union[str, None], list[Document]]
    ) -> dict[str, Union[str, list[Document]]]:
        """Helper function to process a batch of questions."""
        return {
            'content': '\n'.join(f'Q ({question["question_id"]}): {question["question"]}'
                                 for question in questions).strip(),
            'context': get_context_docs(
                docs=grouped_docs,
                must_have_docs=[doc['doc_id']
                                for question in questions
                                for doc in question['docs']],
                model=self._args.model)
        }

    def batch_reason(self, questions: list[QuestionAnswer]) -> list[NoteBook]:
        """
        Dummy batch reasoning since it just returns all the documents in the dataset.

        Args:
            questions (QuestionAnswer): list of questions to reason about
        """
        if not self._index or not self._corpus:
            raise ValueError(
                "Index not created. Please index the dataset before retrieving documents.")

        grouped_docs = {}
        for doc in self._index:
            folder_id = doc.get('folder_id')
            if folder_id:
                grouped_docs.setdefault(folder_id, []).append(doc)
            else:
                grouped_docs.setdefault(None, []).append(doc)

        # Batches questions in groups of 5 that share the same context and
        # ensures all supporting docs are included for all 5 questions
        batch_size = 5

        with Pool(processes=cpu_count()) as pool:
            question_batches = pool.starmap(
                self.process_batch,
                zip([questions[i:i + batch_size]
                     for i in range(0, len(questions), batch_size)], repeat(grouped_docs)),
            )

        Logger().info("Finished processing questions in batch.")

        def get_notebook(question_batch: dict[str, Union[str, list[Document]]]) -> NoteBook:
            """Get the notebook for a question batch.

            Args:
                question_batch (dict[str, Union[str, list[Document]]]): The question batch

            Returns:
                NoteBook: The notebook containing the retrieved documents
            """
            notebook = NoteBook()

            notes = self._qa_prompt.format(
                context=get_content(question_batch['context'])
            )

            notebook.update_notes(notes)
            notebook.update_sources([RetrievedResult(
                doc_id=doc['doc_id'], content=doc['content'], score=None)
                for doc in question_batch['context']])
            notebook.update_questions(question_batch['content'])
            return notebook

        notebooks = [
            get_notebook(question_batch)
            for question_batch in question_batches
        ]

        return notebooks

    def reason(self, _: str) -> NoteBook:
        """Dummy reasoning since it just returns all the documents in the dataset.

        Args:
            _ (str): the question to reason about

        Raises:
            ValueError: if the index is not created

        Returns:
            notebook (NoteBook): the notebook containing the documents
        """
        raise NotImplementedError(
            "Default agent does not support single question reasoning. Use batch_reason instead."
        )


def get_context_docs(
    docs: dict[Union[str, None], list[Document]],
    must_have_docs: list[str],
    model: str
) -> list[Document]:
    """
    Ensures that must_have_docs are included in the final documents.

    Args:
        docs (dict[str, list[Document]]): the documents to be used
        must_have_docs (list[str]): the documents that must be included
        model (str): the model to be used for tokenization

    Returns:
        list[Document]: the final list of documents
    """
    encoding = get_encoding(model)

    max_tokens_map = {
        'gpt-4o-mini': 128_000 * 0.88,
        'gpt-4o-mini-batch': 128_000 * 0.88,
        'o3-mini': 200_000 * 0.88,
    }

    max_tokens = int(max_tokens_map.get(model, 0))

    flattened_docs = [doc for ds in docs.values() for doc in ds]

    start = 0
    end = len(flattened_docs)
    content = get_content(flattened_docs[start:end])

    tokens = encoding.encode(content)

    docs_map = {doc['doc_id']: i for i, doc in enumerate(flattened_docs)}

    if len(tokens) > max_tokens:
        start, end = search_best_interval(
            flattened_docs, must_have_docs, docs_map, max_tokens, encoding)

        content = get_content(flattened_docs[start:end])
        encoded_content = encoding.encode(content)

        if len(encoded_content) > max_tokens:
            content = encoding.decode(encoded_content[:max_tokens])

    return flattened_docs[start:end]

# pylint: disable-next=too-many-locals
def search_best_interval(
    docs: list[Document],
    must_have_docs: list[str],
    docs_map: dict[str, int],
    max_tokens: int,
    encoding: tiktoken.Encoding
) -> tuple[int, int]:
    """
    Find the largest interval (start, end) such that the length of encoded(content)[start, end] is not greater 
    than max_tokens and content is guaranteed to contain all substrings in must_have_docs.
    """
    must_have_indices = [
        docs_map[doc_id] for doc_id in must_have_docs
    ]

    smallest_start = min(start for start in must_have_indices)
    largest_end = max(end for end in must_have_indices)

    def can_extend(start_idx: int, end_idx: int) -> tuple[bool, bool]:
        token_count = len(encoding.encode(
            get_content(docs[start_idx:end_idx])))
        return (token_count <= max_tokens, token_count == max_tokens)

    max_reached = False

    # Binary search to find the maximum extension to the left
    best_left = smallest_start
    left_start, left_end = 0, smallest_start
    while left_start <= left_end:
        mid = (left_start + left_end + 1) // 2
        can_extend_flag, max_r = can_extend(mid, largest_end)
        if not can_extend_flag:
            left_start = mid + 1
        else:
            left_end = mid - 1
            best_left = mid

        if max_r:
            max_reached = True
            break

    best_right = largest_end

    if not max_reached:
        # Binary search to find the maximum extension to the right
        right_start, right_end = largest_end, len(docs)
        while right_start <= right_end:
            mid = (right_start + right_end) // 2
            can_extend_flag, max_r = can_extend(best_left, mid)
            if can_extend_flag:
                best_right = mid
                right_start = mid + 1
            else:
                right_end = mid - 1

            if max_r:
                max_reached = True
                break

    return (best_left, best_right)


def get_content(docs: list[Document]) -> str:
    """
    Get the content of the documents.
    This function assumes that the documents are already sorted by folder_id.
    It will group the documents by folder_id and return the content as a string.
    The content of each document is separated by a newline character.
    The folder_id is included at the beginning of each group of documents.
    The content of each document is also separated by a newline character.

    Args:
        docs (list[Document]): the documents to be used

    Returns:
        str: the content of the documents
    """
    content = []
    cur_folder = None
    for doc in docs:
        if cur_folder != doc['folder_id']:
            content.append(f"sample_id: {doc['folder_id']}")
            cur_folder = doc['folder_id']
        content.append(doc['content'])

    return '\n'.join(content)
