from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.tools.sql_tool import NL2SQLTool
from app.tools.sentiment_tool import SentimentAnalysisTool

class InsightAgent(BaseAgent):
    def __init__(self, llm:LLMClient):
        # 定义insightagent的人设prompt，严格要求输出JSON
        # 在有全局状态管理器的情况下，不再处理sources字段
        role_prompt = (
            "你是专业的舆情与业务分析师 (Insight Agent)。\n"
            "你的任务是综合所有证据，深度分析话题的整体状况。\n"
            "【工具使用强制规定】：\n"
            "1. 如果话题涉及公司产品，你必须调用 query_business_db 查询内部数据。\n"
            "2. 在得出最终的【情感倾向】前，你必须调用 predict_sentiment 工具，将你总结的文本发给它，以获取精准的情感极性和置信度！严禁自行臆测情感。如果你查库花的时间太长，请立刻停止查库，优先调用情感分析工具！\n\n"
            "【输出格式强制要求】：\n"
            "你最后必须且只能输出合法的 JSON 格式，不要包含 Markdown 代码块标签。JSON 必须包含：\n"
            "{\n"
            '  "summary": "一到两句话的高度概括",\n'
            '  "sentiment": "正面/负面/中立/争议 (请附上模型返回的置信度，如：负面，置信度92.5%)",\n'
            '  "key_points": ["要点1", "要点2", "要点3"],\n'
            "}"
        )

        # 实例化SQL工具
        tools = [NL2SQLTool(llm=llm), SentimentAnalysisTool()] 

        # insightagent不需要搜索工具，只做纯文本分析
        # 现在需要用NL2SQL工具
        super().__init__(
            name="InsightAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=tools
        )