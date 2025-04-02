import json
import types
from typing import Any, Dict


###############################################
### json parser class
### Used to parse input json files
class JsonParser():
    def __init__(self) -> None:
        self.pms = None

    def decoder(self, p: Dict[str, Any]) -> types.SimpleNamespace:
        return types.SimpleNamespace(**p)

    def read(self, filename: str) -> types.SimpleNamespace:
        with open(filename, "r") as f:
            self.pms = json.load(f, object_hook=self.decoder)

        return self.pms
