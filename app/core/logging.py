from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

def setup_logger(name: str = "betta") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    return logger

def write_jsonl(path: Union[str, Path], obj:Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"