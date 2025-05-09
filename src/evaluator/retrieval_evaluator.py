
"""Retrieval Evaluator Module."""

from models.document import Document


K_LIST = [1, 2, 5, 10, 20, 100]


def eval_retrieval_recall(doc_pairs: list[tuple[list[Document], list[Document]]]) -> dict[int, float]:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents.

    Args:
        doc_pairs (list[tuple[list[Document], list[Document]]]): \
A list of pairs with the ground documents and the retrieved documents.

    Returns:
        recall_at_k (dict[int, float]): the recall score across various Ks
    """
    recall_at_k = [recall_score(gt, a) for (gt, a) in doc_pairs]

    avg_recall_at_k = {
        k: sum(d[k] for d in recall_at_k) / len(recall_at_k)
        for k in K_LIST
    }

    return avg_recall_at_k


def recall_score(expected_docs: list[Document], actual_docs: list[Document]) -> dict[int, float]:
    """
    Evaluates the recall between the ground truth documents and the model's retrieved documents.

    Args:
        expected_docs (list[Document]): the ground truth documents
        actual_docs (list[Document]): the model's retrieved documents

    Returns:
        recall_at_k (dict[int, float]): the recall score across various Ks
    """
    assert expected_docs, "Expected documents list is empty."
    assert actual_docs, "Actual documents list is empty."

    recall_at_k = {}
    for k in K_LIST:
        top_k_docs = actual_docs[:k]

        correct_at_k = sum(1 for doc in top_k_docs if doc['doc_id'] in [
                           expected_doc['doc_id'] for expected_doc in expected_docs])
        recall_at_k[k] = correct_at_k / len(expected_docs)

    return recall_at_k
