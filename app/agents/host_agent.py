from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient

class HostAgent(BaseAgent):
    def __init__(self, llm:LLMClient):
        role_prompt = (
            "你是这场舆情分析研讨会的主持人 (Host Agent)。\n"
            "你的任务是：阅读前方的【原始证据】以及其他专家的【发言记录】，"
            "客观总结局势，并向下一位发言的专家抛出核心问题。\n"
            "【强制要求】：直接输出纯文本发言，不要输出 JSON。"
        )
        super().__init__(
            name="HostAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=None
        )