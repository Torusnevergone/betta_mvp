from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.tools.search import WebSearchTool
from app.tools.rag_search import LocalRAGTool

class QueryAgent(BaseAgent):
    def __init__(self, llm:LLMClient):
        # 定义 Query Agent 的人设 prompt
        # 在有全局状态管理器的情况下，queryagent不需要输出JSON格式
        role_prompt = (
            "你是专业的舆情情报搜集员 (Query Agent)。\n"
            "你的任务是：根据用户提供的话题，尽可能全面地搜集情报。\n"
            "你有两个强大的工具：\n"
            "1. web_search：用于搜索互联网上的最新新闻和大众舆论。\n"
            "2. local_rag_search：用于搜索公司内部的机密文档、历史预案或私有规定。\n"
            "【行动指南】：\n"
            "- 拿到证据后，请不要做主观情感分析。\n"
            "- 请用清晰的 Markdown 列表，将搜集到的所有客观事实、数据、事件拼接成一个详细的情报清单。\n"
            "- 直接输出纯文本，**绝对不要**输出 JSON 格式。"
        )
        # 实例化它需要的工具
        tools = [WebSearchTool(), LocalRAGTool()]

        # 调用父类初始化
        super().__init__(
            name = "QueryAgent",
            role_prompt = role_prompt,
            llm = llm,
            tools = tools
        )