"""RetrievedResult class."""


class RetrievedResult(dict):
    """
    RetrievedResult class to store the retrieved results.
    It inherits from dict and initializes the dictionary with the given parameters.

    Args:
        dict (Any): dictionary to store the retrieved results
        doc_id (int): the id of the document
        content (str): the content of the document
        score (float): the relevance score of the document
    """

    def __init__(self, doc_id: int, content: str, score: float):
        dict.__init__(self, doc_id=doc_id,
                      content=content, score=score)

    def __repr__(self):
        return f"""RetrievedResult(doc_id={self.get('doc_id')}, content={self.get('content')}),\
    score={self.get('score')})"""
