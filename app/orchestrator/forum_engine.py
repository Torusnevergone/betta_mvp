from __future__ import annotations
import json
import threading
import queue
import re
from typing import Dict, Any
from app.core.llm_client import LLMClient
from app.core.logging import setup_logger, write_jsonl, now_iso
from app.core.state import SessionState
from app.agents.query_agent import QueryAgent
from app.agents.media_agent import MediaAgent
from app.agents.host_agent import HostAgent
from app.agents.insight_agent import InsightAgent
from app.agents.critic_agent import CriticAgent
from app.agents.report_agent import ReportAgent

logger = setup_logger()
def _clean_json_string(raw_str: str) -> str:
    """
    工业级 JSON 提取器。
    使用正则匹配最外层的 {}，无视大模型在前后添加的任何废话。
    """
    # 尝试寻找最外层的 {}
    match = re.search(r'(\{.*\})', raw_str, re.DOTALL)
    if match:
        return match.group(1)
    
    # 如果正则没匹配到，退回到原来的粗暴清理
    cleaned = raw_str.strip()
    if cleaned.startswith("```json"): cleaned = cleaned[7:]
    if cleaned.startswith("```"): cleaned = cleaned[3:]
    if cleaned.endswith("```"): cleaned = cleaned[:-3]
    return cleaned.strip()

class ForumEngine:
    def __init__(self, llm:LLMClient):
        self.llm = llm
        self.trace_path = "app/storage/trace.jsonl"

        # 实例化所有agent
        self.query_agent = QueryAgent(llm=self.llm)
        self.media_agent = MediaAgent(llm=self.llm)
        self.host_agent = HostAgent(llm=self.llm)
        self.insight_agent = InsightAgent(llm=self.llm)
        self.critic_agent = CriticAgent(llm=self.llm)
        self.report_agent = ReportAgent(llm=self.llm)

    # 加入db后，增加history参数
    def run(self, topic:str, history:list = None):
        """
        使用多线程和队列模拟真正的事件驱动论坛机制。
        """
        # 引入上下文
        if history is None:
            history = []
        # 把历史记录格式转化为纯文本，塞进Agent的背景信息
        history_text = ""
        if history:
            history_text = "【之前的对话上下文】:\n" + "\n".join([f"{m.role}: {m.content[:200]}..." for m in history]) + "\n\n"

        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_start", "topic": topic})
        logger.info(f"[ForumEngine] 启动事件驱动论坛，议题: {topic}")

        state = SessionState(topic=topic)

        # 1. 第一阶段：多线程并发搜集情报
        logger.info("\n>>> [阶段一] 多线程并发情报搜集 <<<")
        # 用于接收线程结果的容器
        evidence_results = {"query": "", "media": ""}
        # 定义线程任务函数
        # 加入db，把历史上下文加进 Query 和 Media 的 Prompt 里
        def fetch_query():
            # prompt = f"请帮我搜集关于话题：【{topic}】的最新情报。"
            prompt = f"{history_text}请结合上述上下文，帮我搜集关于最新话题：【{topic}】的情报。"
            evidence_results["query"] = self.query_agent.chat(prompt, state=state)
        def fetch_media():
            # prompt = f"请帮我搜集关于话题：【{topic}】的短视频平台最新情报。"
            prompt = f"{history_text}请结合上述上下文，帮我搜集关于最新话题：【{topic}】的短视频平台情报。"
            evidence_results["media"] = self.media_agent.chat(prompt, state=state)

        # 创建并启动线程
        t1 = threading.Thread(target=fetch_query, name="Thread-Query")
        t2 = threading.Thread(target=fetch_media, name="Thread-Media")
        t1.start()
        t2.start()

        # 阻塞主线程，等待两个探员都回来
        t1.join()
        t2.join()

        raw_evidence_text = f"【图文新闻与内部文档】:\n{evidence_results['query']}\n\n【短视频与多模态】:\n{evidence_results['media']}"
        logger.info("[ForumEngine] 情报搜集完毕，进入会议室。")

        # 2. 第二阶段：基于消息循环的论坛辩论
        logger.info("\n>>> [阶段二] 论坛辩论开始 <<<")
        max_rounds = 2
        current_round = 1
        final_insight_data = {}

        while current_round <= max_rounds:
            logger.info(f"--- 第 {current_round} 轮辩论 ---")

            # Host发言
            host_prompt = f"这是最新证据：\n{raw_evidence_text}\n\n请作为主持人，向 Insight Agent 抛出争议点。"
            host_speech = self.host_agent.chat(host_prompt)
            state.add_chat_record(self.host_agent.name, host_speech)
            logger.info(f"[主持人]: {host_speech}")
            # Insight发言
            insight_prompt = (
                f"这是原始证据：\n{raw_evidence_text}\n\n"
                f"这是论坛记录：\n{state.get_forum_context()}\n\n"
                f"请回应主持人。如果你调用了内部数据库，请在分析报告中明确指出数据来源。\n"
                f"请严格输出 JSON 分析结果。"
            )
            insight_result_str = self.insight_agent.chat(insight_prompt)
            # 【修复点】：不要只写一句废话，把 Insight 思考过程中调用工具查到的信息也记在黑板上！
            # 虽然我们拿不到它查库的原始文本，但我们可以把它生成的 summary 贴出来给大家看。
            # state.add_chat_record(self.insight_agent.name, "我已提交深度分析报告。")
            try:
                insight_data = json.loads(_clean_json_string(insight_result_str))
                final_insight_data = insight_data
                # 把它的核心结论公开，这样 Critic 就不会觉得它是凭空捏造的了
                state.add_chat_record(self.insight_agent.name, f"我已结合内外数据完成分析，核心结论：{insight_data.get('summary', '')}")
            except json.JSONDecodeError:
                logger.error("Insight JSON 解析失败")
                break
            # Critic发言
            critic_prompt = (
                f"请审核 Insight 的报告。\n"
                f"【外部原始证据】\n{raw_evidence_text}\n"
                f"【论坛聊天记录（包含内部数据查询结果）】\n{state.get_forum_context()}\n"
                f"【分析报告】\n{json.dumps(insight_data, ensure_ascii=False)}\n\n"
                f"请判断报告是否合格。注意：报告中的数据可能来源于【外部原始证据】或【论坛聊天记录】中提到的内部数据库查询，只要在其中一处有依据，即不算幻觉。\n"
                f"严格输出 JSON 格式。"
            )
            critic_result_str = self.critic_agent.chat(critic_prompt)
            try:
                critic_data = json.loads(_clean_json_string(critic_result_str))
                is_passed = critic_data.get("passed", False)
                feedback = critic_data.get("feedback", "无反馈")
                state.add_chat_record(self.critic_agent.name, f"审核意见：{'通过' if is_passed else '驳回'}。理由：{feedback}")
                logger.info(f"[Critic]: {'通过' if is_passed else '驳回'} - {feedback}")
                if is_passed:
                    break
                else:
                    if current_round < max_rounds:
                        state.add_chat_record(self.host_agent.name, "专家们存在分歧，请 Insight 重新审视证据并修改报告！")
                    else:
                        # 【新增修复】：明确打印达到最大轮数的警告，解释为什么直接去写报告了
                        logger.warning(f"[ForumEngine] 已达到最大辩论轮数 ({max_rounds})，尽管 Critic 驳回，仍将强制采用当前版本的报告。")
                        break
            except json.JSONDecodeError:
                break
            current_round += 1

        # 3. 阶段三：主编撰写最终报告
        final_data_str = json.dumps({
            "topic": topic,
            "sentiment": final_insight_data.get('sentiment','未知'),
            "summary": final_insight_data.get('summary','暂无摘要数据'),
            "key_points": final_insight_data.get('key_points', []),
            "sources": state.sources 
        }, ensure_ascii=False, indent=2)
        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_end", "topic": topic})
        
        # 加入GraphRAG后，把 state.forum_logs 传进去，供 GraphRAG 画图使用
        # return self.report_agent.generate_stream(final_data_str)
        return self.report_agent.generate_stream(final_data_str, forum_logs=state.forum_logs)

            


