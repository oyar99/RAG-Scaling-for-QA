"""RetrievedResult class."""

class RetrievedResult(dict):
    """A RetrievedResult class that inherits from dict.

    Args:
        dict: inherits from dict
    """

    def __init__(self, doc_id: int, content: str, score: float):
        dict.__init__(self, doc_id=doc_id,
                      content=content, score=score)

    def __repr__(self):
        return f"""RetrievedResult(doc_id={self.get('doc_id')}, content={self.get('content')}),\
    score={self.get('score')})"""
    