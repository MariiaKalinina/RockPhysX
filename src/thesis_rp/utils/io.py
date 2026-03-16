from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path


def write_json(path: str | Path, obj) -> None:
    """Serialize a dataclass or plain mapping to JSON."""
    payload = asdict(obj) if is_dataclass(obj) else obj
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
