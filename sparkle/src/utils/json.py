import json
import types
from typing import Any, Dict


###############################################
### json parser class
### Used to parse input json files
class JsonParser():
    """
    JSON parser class.

    This class provides methods for reading and parsing JSON files.
    """
    def __init__(self) -> None:
        """
        Initializes the JsonParser.
        """
        self.pms = None

    def decoder(self, p: Dict[str, Any]) -> types.SimpleNamespace:
        """
        Decodes a dictionary into a SimpleNamespace object.

        Args:
            p: The dictionary to decode.

        Returns:
            A SimpleNamespace object containing the dictionary's data.
        """
        return types.SimpleNamespace(**p)

    def read(self, filename: str) -> types.SimpleNamespace:
        """
        Reads and parses a JSON file.

        Args:
            filename: The path to the JSON file.

        Returns:
            A SimpleNamespace object containing the parsed JSON data.
        """
        with open(filename, "r") as f:
            self.pms = json.load(f, object_hook=self.decoder)

        return self.pms
