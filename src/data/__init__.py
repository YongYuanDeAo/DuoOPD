"""Dataset construction and shared input helpers."""


def plain_nested(value):
    """Convert Arrow/NumPy nested values into ordinary Python containers."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): plain_nested(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_nested(item) for item in value]
    return value
