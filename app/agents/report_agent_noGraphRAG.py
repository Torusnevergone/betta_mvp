from __future__ import annotations
from typing import Generator
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.core.types import Message

class ReportAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        role_prompt = (
            "你是专业的舆情报告主编 (Report Agent)。\n"
            "你的任务是：接收前面分析师整理好的结构化数据（JSON格式），"
            "并将其扩写、排版成一篇专业、流畅、排版精美的 Markdown 舆情报告。\n"
            "【强制要求】：\n"
            "1. 报告必须包含：标题、执行摘要、情感倾向、关键发现（展开细说）、参考来源列表。\n"
            "2. 在【参考来源列表】中，必须严格按照 '[序号]. [标题](URL)' 的 Markdown 链接格式输出所有来源。\n" # <--- 新增这句指令
            "3. 语言要专业、客观，像一份提交给高管的商业报告。\n"
            "4. 严禁篡改数据，必须严格基于输入的数据进行扩写。\n"
            "5. 直接输出 Markdown 文本，不要用 ```markdown 包裹。"
        )

        super().__init__(
            name="ReportAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=None
        )

    def generate_stream(self, structured_data: str) -> Generator[str, None, None]:
        """
        流式生成报告的专属方法。
        它直接调用 llm_client 的 chat_stream 方法，不走复杂的 Tool Calling 循环。
        """
        messages = [
            Message(role="system", content=self.role_prompt),
            Message(role="user", content=f"请基于以下数据生成最终的舆情报告：\n\n{structured_data}")
        ]

        print(f"\n[{self.name}] 正在奋笔疾书撰写最终报告...\n")
        # 直接透传底层LLM的流式生成器
        for chunk in self.llm.chat_stream(messages=messages, tools=None):
            # 防呆过滤，确保chunk不是None，且不是纯空字符串
            if chunk is not None and str(chunk) != "":
                text_chunk = str(chunk)
                print(text_chunk, end="", flush=True) # 在终端实时打印
                # yield str(chunk)
                yield chunk
    