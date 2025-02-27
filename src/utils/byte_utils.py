def format_size(size):
    for unit in ['B', 'K', 'M', 'G']:
        if size < 1024.0:
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}T"
