from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient

class CriticAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        # 设定严厉人设
        role_prompt = (
            "你是专业的舆情质量校验员 (Critic Agent)。\n"
            "你的任务是审核【分析报告】是否合格，并检查其是否有充足的【原始证据】支撑。\n"
            "你不需要调用工具。请严格按照以下标准进行审核：\n"
            "1. 若报告声明'证据不足'或'未找到信息'，判定为不合格。\n"
            "2. 若报告的结论在原始证据中无明确出处（存在幻觉），判定为不合格。\n"
            "3. 若结论缺乏具体的数据或事件支撑，判定为不合格。\n\n"
            "【强制要求】：你必须且只能输出合法的 JSON 格式，不要包含 Markdown 代码块标签（如 ```json）。JSON 必须包含以下字段：\n"
            "{\n"
            '  "passed": true 或 false, // 审核是否通过\n'
            '  "feedback": "如果不通过，请简要指出报告的缺陷",\n'
            '  "suggested_queries": ["如果需要补充检索，请提供1-2个更精准的搜索关键词"] // 若 passed 为 true，可返回空列表\n'
            "}"
        )
        
        super().__init__(
            name="CriticAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=None
        )