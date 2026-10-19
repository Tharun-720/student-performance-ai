"""Small I/O helpers."""
import json
from pathlib import Path
from typing import Any, Union


def read_json(path: Union[str, Path]) -> Any:
    """Read a UTF-8 JSON file and close the handle."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
