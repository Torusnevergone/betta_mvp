from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Optional, List, Dict

# 增加"tool"角色
Role = Literal["system", "user", "assistant", "tool"]

@dataclass
class Message:
    role: Role
    content: str
    # 行业规范：当 role 为 "assistant" 且调用工具时，需要这个字段记录它想调什么工具
    tool_calls: Optional[list[Dict[str, Any]]] = None
    # 行业规范：当 role 为 "tool" 时，必须带上这个 ID，告诉模型这是哪个工具的结果
    tool_call_id: Optional[str] = None
    # 工具名称
    name: Optional[str] = None

@dataclass
class Evidence:
    source: str
    title: str
    url: str
    snippet: str
    meta: Optional[dict[str,Any]] = None