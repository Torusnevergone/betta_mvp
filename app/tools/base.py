from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.core.state import SessionState

class BaseTool(ABC):
    """
    所有工具的抽象基类
    """

    name: str
    description: str
    parameters: Dict[str, Any]

    @abstractmethod
    def run(self, state: Optional[SessionState], **kwargs) -> str:
        """
        【架构升级】：增加 state 参数。
        工具在执行时，可以直接修改 state（产生副作用），
        同时返回一个精简的文本给 LLM（避免 Token 浪费）。
        即：工具在执行时，不仅返回文本给大模型看，还要直接把结构化数据（如 URL）写入 State。
        """
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """
        转换成 OpenAI tools 格式
        """
        return{
            "type":"function",
            "function":{
                "name":self.name,
                "description":self.description,
                "parameters":self.parameters,
            },
        }
