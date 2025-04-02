from typing import Any, Callable

from sparkle.src.utils.error import error


class Factory:
    """
    A factory class for creating objects based on a key.

    This class provides a mechanism to register object creators (e.g., class
    constructors) with string keys and then instantiate objects of the
    corresponding type using those keys.

    Attributes:
        keys (dict): A dictionary mapping keys to object creators.
    """

    def __init__(self):
        """
        Initializes the factory.
        """
        self.keys = {}

    def register(self, key: str, creator: Callable[[str], Any]):
        """
        Registers a key/creator pair.

        Args:
            key: The key to register.
            creator: The class constructor or callable associated with the key.
        """
        self.keys[key] = creator

    def create(self, key: str, **kwargs) -> Any:
        """
        Instantiates and returns an object of the type associated with the key.

        Args:
            key: The key to look up in self.keys.
            **kwargs: Arbitrary keyword arguments for the constructor.

        Returns:
            An instance of the type defined by the key.

        Raises:
            error: If an unknown key is provided.
        """
        creator = self.keys.get(key)
        if not creator:
            error("factory", "create", f"Unknown key provided: {key}")

        return creator(**kwargs)
