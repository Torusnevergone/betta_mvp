from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.tools.sql_tool import NL2SQLTool

class InsightAgent(BaseAgent):
    def __init__(self, llm:LLMClient):
        # 定义insightagent的人设prompt，严格要求输出JSON
        # 在有全局状态管理器的情况下，不再处理sources字段
        role_prompt = (
            "你是专业的舆情与业务分析师 (Insight Agent)。\n"
            "你的任务是：综合前方探员提供的【外部舆情证据】以及你通过工具查询到的【内部业务数据】，深度分析该话题的整体状况。\n"
            "你拥有 query_business_db 工具，如果话题涉及公司产品（如小米SU7），你必须调用该工具查询内部的客诉和销售数据，来验证外部舆情是否属实。\n"
            "【强制要求】：你最后必须且只能输出合法的 JSON 格式，不要包含任何 Markdown 代码块标签（如 ```json）。JSON 必须包含以下字段：\n"
            "{\n"
            '  "summary": "一到两句话的高度概括，必须结合内外部数据",\n'
            '  "sentiment": "正面/负面/中立/争议",\n'
            '  "key_points": ["要点1", "要点2", "要点3"],\n'
            "}"
        )

        # 实例化SQL工具
        tools = [NL2SQLTool(llm=llm)] 

        # insightagent不需要搜索工具，只做纯文本分析
        # 现在需要用NL2SQL工具
        super().__init__(
            name="InsightAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=tools
        )