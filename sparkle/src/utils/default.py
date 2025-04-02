###############################################
### Helper function to set default value if
### user-specified value is not present
from types import SimpleNamespace
from typing import Any, Optional

def set_default(name: str, default_value: Any, pms: SimpleNamespace) -> Any:

    if (hasattr(pms, name)): return getattr(pms, name)
    else: return default_value
