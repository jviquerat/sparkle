from types import SimpleNamespace
from typing import Any, Optional


def set_default(name: str, default_value: Any, pms: SimpleNamespace) -> Any:
    """
    Sets a default value for a parameter if it is not present in the parameter namespace.

    Args:
        name: The name of the parameter.
        default_value: The default value to use if the parameter is not found.
        pms: The SimpleNamespace object containing the parameters.

    Returns:
        The value of the parameter, either the user-specified value or the default value.
    """

    if (hasattr(pms, name)): return getattr(pms, name)
    else: return default_value
