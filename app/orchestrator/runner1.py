from __future__ import annotations
from app.core.llm_client import LLMClient
from app.core.types import Message
from app.core.logging import write_jsonl, now_iso, setup_logger
from app.report.schema import ReportIR

import json

from app.agents.query_agent import QueryAgent
from app.agents.insight_agent import InsightAgent

logger = setup_logger()

class Runner:
    def __init__(self, llm: LLMClient, trace_path: str = "app/storage/trace.jsonl"):
        self.llm = llm
        self.trace_path = trace_path

        # 实例化第一个agent
        self.query_agent = QueryAgent(llm=self.llm)
        # 实例化第二个agent
        self.insight_agent = InsightAgent(llm=self.llm)

    def run(self, topic: str) -> ReportIR:
        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_start", "topic": topic})

        logger.info(f"启动多agent舆情分析，当前话题：{topic}")

        # 阶段一：让 Query Agent 自己去干活
        logger.info("\n>>> [阶段一] 情报搜集开始 <<<")
        # 只有queryagent的时候：我们只给它一句话，它会自己决定调用 search 工具，然后把结果总结给我们
        # query_result = self.query_agent.chat(f"请帮我搜集关于话题：【{topic}】的最新情报。")

        query_prompt = f"请帮我搜集关于话题：【{topic}】的最新情报。"
        raw_evidence = self.query_agent.chat(query_prompt)
        logger.info("Query Agent搜集完毕。")

        # print("\n Query Agent 的最终汇报：\n")
        # print(query_result)
        # TODO: 这里目前只是把 Query Agent 的输出直接塞进了报告。
        # 之后我们会在这里引入 Insight Agent 和 Critic Agent，
        # 把 query_result 交给它们去写深度 JSON 并进行论坛辩论。

        # 阶段二：Insight Agent负责深度分析并输出JSON
        logger.info("\n>>> [阶段二] 深度分析开始 <<<")
        insight_prompt = f"这是关于话题【{topic}】的原始证据清单：\n\n{raw_evidence}\n\n请严格按照系统设定的 JSON 格式输出分析结果。"
        insight_result_str = self.insight_agent.chat(insight_prompt)
        logger.info("Insight Agent 分析完毕。原始输出如下：")
        logger.info(insight_result_str)

        # 阶段三：解析JSON并声称标准化IR报告
        # 因为大模型有时候会不听话加上 ```json 标签，我们做个简单的清理
        cleaned_str = insight_result_str.strip()
        if cleaned_str.startswith("```json"):
            cleaned_str = cleaned_str[7:]
        if cleaned_str.startswith("```"):
            cleaned_str = cleaned_str[3:]
        if cleaned_str.endswith("```"):
            cleaned_str = cleaned_str[:-3]
        
        try:
            insight_data = json.loads(cleaned_str.strip())
        except json.JSONDecodeError:
            logger.error("Insight Agent 输出的不是合法的 JSON！降级处理。")
            # 降级处理：如果解析失败，就把它当成纯文本塞进去
            insight_data = {
                "summary": "解析失败，原始内容：" + insight_result_str,
                "sentiment": "未知",
                "key_points":["解析失败"]
            }
        
        '''
        resp = self.llm.chat([
            Message(role="system", content="你是舆情分析助手。"),
            Message(role="user", content=f"给我一句话概括：{topic}"),
        ])
        '''

        # 组装最终的 ReportIR
        # 注意：这里展示了如何把 Insight Agent 提炼的 key_points 传给 schema
        ir = ReportIR(
            topic=topic, 
            # summary=query_result,   # 暂时用queryagent汇报顶替
            summary=f"【情感倾向：{insight_data.get('sentiment', '未知')}】 {insight_data.get('summary', '')}",
            key_points=insight_data.get("key_points", []),
            sources = []            # 暂时留空
            )

        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_end", "topic": topic})
        return ir