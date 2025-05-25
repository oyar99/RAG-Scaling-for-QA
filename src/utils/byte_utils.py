"""A module to format a size in bytes into a human-readable string."""


def format_size(size: float) -> str:
    """
    Formats a integer representing a size in bytes into a human-readable string.
    For example, 1024 bytes will be formatted as "1.00KB".
    It supports representing sizes in bytes, kilobytes, megabytes, and gigabytes.
    The size is rounded to two decimal places and the appropriate unit is appended.

    Args:
        size (float): the size in bytes

    Returns:
        formatted_size (str): a human-readable string representation of the size
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}T"
