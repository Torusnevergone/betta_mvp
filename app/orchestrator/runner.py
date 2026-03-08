from __future__ import annotations
from app.core.llm_client import LLMClient
from app.core.types import Message
from app.core.logging import write_jsonl, now_iso, setup_logger
from app.report.schema import ReportIR

import json
import re
from typing import Dict, Any, List
import asyncio

from app.agents.query_agent import QueryAgent
from app.agents.insight_agent import InsightAgent
from app.agents.critic_agent import CriticAgent
from app.agents.report_agent import ReportAgent
from app.agents.media_agent import MediaAgent

from app.core.state import SessionState

logger = setup_logger()

# 辅助函数：清理大模型可能带上的 ```json 标签
def _clean_json_string(raw_str: str) -> str:
    """
    辅助函数：清理大模型输出时可能附带的 Markdown 代码块标签。
    这是处理 LLM 结构化输出时常见的防御性措施。
    """
    cleaned = raw_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

'''
# 行业规范的做法：在 Runner 层全局拦截与提取
# 既然 Runner（项目经理）能看到所有人的聊天记录，那我们就在 Runner 里写一个简单的正则（或字符串解析），
# 把 QueryAgent 搜回来的原始证据（raw_evidence）里的标题和 URL 抠出来，直接塞给最终的报告。
def _extract_sources(raw_evidence: str) -> List[Dict[str, str]]:
    """
    辅助函数：从 Query Agent 的原始输出中提取来源链接。
    因为我们的 WebSearchTool 返回的格式包含 'title' 和 'url'，
    我们可以用简单的正则来把它们捞出来。
    """
    sources = []
    # 匹配类似 "url": "https://..." 或 'url': 'https://...'
    url_pattern = r'"url"\s*:\s*"([^"]+)"|\'url\'\s*:\s*\'([^\']+)\''
    title_pattern = r'"title"\s*:\s*"([^"]+)"|\'title\'\s*:\s*\'([^\']+)\''
    urls = [match[0] or match[1] for match in re.findall(url_pattern, raw_evidence)]
    titles = [match[0] or match[1] for match in re.findall(title_pattern, raw_evidence)]
    # 组合 title 和 url，去重
    seen_urls = set()
    for i in range(min(len(urls), len(titles))):
        if urls[i] not in seen_urls and urls[i].startswith("http"):
            sources.append({
                "title": titles[i],
                "url": urls[i]
            })
            seen_urls.add(urls[i])
            
    return sources
'''

def _extract_sources_to_state(raw_json_str: str, state: SessionState):
    """
    加入全局状态存储器后，从baseagent发给LLM的messages中提取URL
    辅助函数：从工具返回的 JSON 字符串中提取 URL，并写在黑板上。
    因为工具的 run 方法返回的是标准的 JSON 字符串列表。
    """
    try:
        results = json.loads(raw_json_str)
        if isinstance(results, list):
            for item in results:
                # 写入全局状态黑板
                state.add_evidence(
                    text=item.get("snippet",""),
                    source_title=item.get("title",""),
                    source_url=item.get("url","")
                )
    except json.JSONDecodeError:
        pass # 如果不是JSON，说明不是工具返回的，忽略


class Runner:
    def __init__(self, llm: LLMClient, trace_path: str = "app/storage/trace.jsonl"):
        self.llm = llm
        self.trace_path = trace_path

        # 实例化第一个agent
        self.query_agent = QueryAgent(llm=self.llm)
        # 实例化第二个agent
        self.insight_agent = InsightAgent(llm=self.llm)
        # 实例化第三个agent
        self.critic_agent = CriticAgent(llm=self.llm)
        # 实例化第四个agent
        self.report_agent = ReportAgent(llm=self.llm)

        self.media_agent = MediaAgent(llm=self.llm)

    # 恢复为同步方法
    def run(self, topic: str):
        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_start", "topic": topic})
        logger.info(f"[Runner] 启动多 Agent 协作流程，目标话题: {topic}")
        state = SessionState(topic=topic)
        
        max_rounds = 2
        current_round = 1
        current_query_prompt = f"请帮我搜集关于话题：【{topic}】的最新情报。"
        final_insight_data: dict[str, any] = {}
        while current_round <= max_rounds:
            logger.info(f"[Runner] 开始第 {current_round}/{max_rounds} 轮迭代")
            logger.info("\n>>> [阶段一] 并发情报搜集开始 <<<")
            
            # 使用 asyncio.run 在同步函数中执行异步并发！
            import asyncio
            
            async def run_concurrent_agents():
                async def run_query():
                    logger.info("[Runner] 派遣 Query Agent 搜集图文新闻...")
                    return await asyncio.to_thread(self.query_agent.chat, current_query_prompt, None, state)
                async def run_media():
                    logger.info("[Runner] 派遣 Media Agent 搜集短视频情报...")
                    media_prompt = f"请帮我搜集关于话题：【{topic}】的短视频平台最新情报。"
                    return await asyncio.to_thread(self.media_agent.chat, media_prompt, None, state)
                # 并发执行并等待结果
                return await asyncio.gather(run_query(), run_media())
            # 魔法：同步等待异步并发完成
            results = asyncio.run(run_concurrent_agents())
            query_evidence = results[0]
            media_evidence = results[1]
            raw_evidence_text = f"【图文新闻与内部文档证据】:\n{query_evidence}\n\n【短视频与多模态证据】:\n{media_evidence}"
            logger.info("[Runner] 所有 Agent 并发搜集完毕。")
            # 阶段二：Insight Agent分析 (保持不变)
            logger.info("\n>>> [阶段二] 深度分析开始 <<<")
            insight_prompt = f"这是关于话题【{topic}】的原始证据清单：\n\n{raw_evidence_text}\n\n请严格按照系统设定的 JSON 格式输出分析结果。"
            insight_result_str = self.insight_agent.chat(insight_prompt)
            logger.info(insight_result_str)
            
            try:
                insight_data = json.loads(_clean_json_string(insight_result_str))
                final_insight_data = insight_data
            except json.JSONDecodeError:
                break
                
            # 阶段三：Critic Agent校验 (保持不变)
            logger.info("\n>>> [阶段三] 质量校验开始 <<<")
            critic_prompt = (
                f"请审核以下分析报告。\n"
                f"【原始证据】\n{raw_evidence_text}\n\n"
                f"【分析报告】\n{json.dumps(insight_data, ensure_ascii=False)}\n\n"
                f"请判断报告是否合格，并严格按照 JSON 格式输出审核结果。"
            )
            critic_result_str = self.critic_agent.chat(critic_prompt)
            
            try:
                critic_data = json.loads(_clean_json_string(critic_result_str))
                if critic_data.get("passed", False):
                    break
                else:
                    if current_round < max_rounds and critic_data.get("suggested_queries", []):
                        current_query_prompt = f"之前搜集的情报不足。请重点搜索：\n{', '.join(critic_data.get('suggested_queries', []))}\n目标话题：【{topic}】"
                    else:
                        break
            except json.JSONDecodeError:
                break
            
            current_round += 1
        final_data_str = json.dumps({
            "topic": topic,
            "sentiment": final_insight_data.get('sentiment','未知'),
            "summary": final_insight_data.get('summary','暂无摘要数据'),
            "key_points": final_insight_data.get('key_points', []),
            "sources": state.sources 
        },ensure_ascii=False, indent=2)
        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_end", "topic": topic})
        
        # 恢复普通的 return 生成器
        return self.report_agent.generate_stream(final_data_str)