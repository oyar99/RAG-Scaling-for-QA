"""A module to format a size in bytes into a human-readable string."""
def format_size(size: int) -> str:
    """Formats a intenger representing a size in bytes into a human-readable string.

    Args:
        size (int): the size in bytes

    Returns:
        str: a human-readable string representation of the size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}T"
