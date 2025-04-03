"""Utilities for models."""

def supports_temperature_param(model: str) -> bool:
    """
    Checks if the model supports temperature parameter.

    Args:
        model (str): the model identifier

    Returns:
        bool: True if the model supports temperature parameter, False otherwise
    """
    # Models that support temperature parameter
    models_with_temp = [
        'gpt-4o-mini',
    ]

    return model in models_with_temp
