"""Document class."""


class Document(dict):
    """
    Document class to store the document information.
    It inherits from dict and initializes the dictionary with the given parameters.

    Args:
        dict (Any): dictionary to store the document information
        doc_id (int): the id of the document
        content (str): the content of the document
    """

    def __init__(self, doc_id: int, content: str):
        dict.__init__(self, doc_id=doc_id,
                      content=content)

    def __repr__(self):
        return f"""Document(doc_id={self.get('doc_id')}, content={self.get('content')})"""
