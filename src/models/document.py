"""Document class."""

class Document(dict):
    """A Document class that inherits from dict.

    Args:
        dict: inherits from dict
    """

    def __init__(self, doc_id: int, content: str):
        dict.__init__(self, doc_id=doc_id,
                      content=content)

    def __repr__(self):
        return f"""Document(doc_id={self.get('doc_id')}, content={self.get('content')})"""
    