from typing import Any, Callable

from sparkle.src.utils.error import error


###############################################
class Factory:
    """
    A factory class

    This class is used to generate objects of given type based on a type string

    Attributes:
        keys (dict): dict containing registered key/creator pairs
    """

    def __init__(self):
        """
        Initializes the factory
        """
        self.keys = {}

    def register(self, key: str, creator: Callable[[str], Any]):
        """
        Register a key/creator pair

        Args:
            key: key to register
            creator: class constructor associated to key
        """
        self.keys[key] = creator

    def create(self, key: str, **kwargs) -> Any:
        """
        Instantiate and return an object of the type associated to key

        Args:
            key: key to look up in self.keys
            **kwargs: arbitrary keyword arguments for the constructor

        Returns:
            creator(**kwargs): instance of type defined by key
        """
        creator = self.keys.get(key)
        if not creator:
            error("factory", "create", f"Unknown key provided: {key}")

        return creator(**kwargs)
