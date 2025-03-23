"""Hash utilities."""
import hashlib


def get_content_hash(content: str) -> str:
    """
    Gets the content unique hash.

    Args:
        question (str): the content to be hashed

    Returns:
        content_hash (str): the content hash
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content.encode('utf-8'))
    return sha256_hash.hexdigest()
