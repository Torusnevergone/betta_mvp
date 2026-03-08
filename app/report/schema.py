from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List

class ReportIR(BaseModel):
    topic: str
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)