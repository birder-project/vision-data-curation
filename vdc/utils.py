import json
from typing import Any


def read_json(json_path: str) -> dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)

    return data
