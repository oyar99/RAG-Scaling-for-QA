"""Document class."""


class Document(dict):
    """
    Document class to store the document information.
    It inherits from dict and initializes the dictionary with the given parameters.

    Args:
        dict (Any): dictionary to store the document information
        folder_id (str): the id of the parent folder if any
        doc_id (str): the id of the document
        content (str): the content of the document
    """

    def __init__(self, doc_id: str, folder_id: str, content: str):
        dict.__init__(self, doc_id=doc_id, folder_id=folder_id,
                      content=content)

    def __repr__(self):
        return f"""Document(doc_id={self.get('doc_id')}, \
folder_id={self.get('folder_id')} content={self.get('content')})"""
