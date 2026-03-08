from __future__ import annotations
from app.core.llm_client import LLMClient
from app.core.types import Message
from app.core.logging import write_jsonl, now_iso, setup_logger
from app.report.schema import ReportIR

import json
import re
from typing import Dict, Any, List

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

    def run(self, topic: str) -> ReportIR:
        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_start", "topic": topic})

        # logger.info(f"启动多agent舆情分析，当前话题：{topic}")
        logger.info(f"[Runner] 启动多 Agent 协作流程，目标话题: {topic}")

        # 实例化本次对话的全局黑板
        state = SessionState(topic=topic)

        # 控制变量
        max_rounds = 2 # 最多允许打回重做 1 次（总共跑 2 轮）
        current_round = 1
        # 初始的查询提示词
        current_query_prompt = f"请帮我搜集关于话题：【{topic}】的最新情报。"
        
        # 用于保存最后一次成功解析的数据,存储最终采纳的分析数据
        final_insight_data: dict[str, any] = {}
        # 用于保存最终被采纳的那一轮的原始证据
        final_raw_evidence = ""

        while current_round <= max_rounds:
            logger.info(f"[Runner] 开始第 {current_round}/{max_rounds} 轮迭代")

            # 阶段一：让 Query Agent 自己去干活
            logger.info("\n>>> [阶段一] 情报搜集开始 <<<")
            # 只有queryagent的时候：我们只给它一句话，它会自己决定调用 search 工具，然后把结果总结给我们
            # query_result = self.query_agent.chat(f"请帮我搜集关于话题：【{topic}】的最新情报。")
            
            # raw_evidence = self.query_agent.chat(current_query_prompt)

            # 核心：把state传给QueryAgent
            # queryagent在内部调用工具时，工具会把URL写进这个state里
            # raw_evidence_text = self.query_agent.chat(current_query_prompt, state=state)
            # 1. 广度搜索 (Query Agent)
            logger.info("[Runner] 派遣 Query Agent 搜集图文新闻...")
            query_evidence = self.query_agent.chat(current_query_prompt, state=state)
            # 2. 多模态搜索 (Media Agent) - 新增
            logger.info("[Runner] 派遣 Media Agent 搜集短视频情报...")
            media_prompt = f"请帮我搜集关于话题：【{topic}】的短视频平台最新情报。"
            media_evidence = self.media_agent.chat(media_prompt, state=state)
            # 3. 汇总所有原始证据
            raw_evidence_text = f"【图文新闻与内部文档证据】:\n{query_evidence}\n\n【短视频与多模态证据】:\n{media_evidence}"


            # 暂存当前轮次的证据
            # final_raw_evidence = raw_evidence

            # query_prompt = f"请帮我搜集关于话题：【{topic}】的最新情报。"
            logger.info("Query Agent搜集完毕。")

            # print("\n Query Agent 的最终汇报：\n")
            # print(query_result)
            # TODO: 这里目前只是把 Query Agent 的输出直接塞进了报告。
            # 之后我们会在这里引入 Insight Agent 和 Critic Agent，
            # 把 query_result 交给它们去写深度 JSON 并进行论坛辩论。

            # 阶段二：Insight Agent负责深度分析并输出JSON
            logger.info("\n>>> [阶段二] 深度分析开始 <<<")
            insight_prompt = f"这是关于话题【{topic}】的原始证据清单：\n\n{raw_evidence_text}\n\n请严格按照系统设定的 JSON 格式输出分析结果。"
            insight_result_str = self.insight_agent.chat(insight_prompt)
            logger.info("Insight Agent 分析完毕。原始输出如下：")
            logger.info(insight_result_str)

            try:
                insight_data = json.loads(_clean_json_string(insight_result_str))
                final_insight_data = insight_data # 暂存当前轮次结果
            except json.JSONDecodeError:
                logger.error("[Runner] InsightAgent 输出解析失败，终止当前迭代。")
                break


            # 阶段三：解析JSON并声称标准化IR报告          
            '''
            # 因为大模型有时候会不听话加上 ```json 标签，我们做个简单的清理
            cleaned_str = insight_result_str.strip()
            if cleaned_str.startswith("```json"):
                cleaned_str = cleaned_str[7:]
            if cleaned_str.startswith("```"):
                cleaned_str = cleaned_str[3:]
            if cleaned_str.endswith("```"):
                cleaned_str = cleaned_str[:-3]
            '''
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
                is_passed = critic_data.get("passed", False)
                feedback = critic_data.get("feedback", "无反馈")
                suggested_queries = critic_data.get("suggested_queries", [])

                if is_passed:
                    logger.info("[CriticAgent] 校验通过。")
                    break
                else:
                    logger.warning(f"[CriticAgent] 校验未通过。反馈: {feedback}")
                    # 检查是否允许进入下一轮重试
                    if current_round < max_rounds and suggested_queries:
                        logger.info(f"[Runner] 准备基于建议进行重试，新检索词: {suggested_queries}")
                        # 根据 Critic 的建议，动态构造下一轮的检索指令
                        current_query_prompt = (
                            f"之前搜集的情报不足或偏离主题。请重点使用以下关键词进行深度搜索：\n"
                            f"{', '.join(suggested_queries)}\n"
                            f"目标话题：【{topic}】"
                        )
                    else:
                        logger.warning("[Runner] 已达最大迭代轮数，将采用当前报告。")
            except json.JSONDecodeError:
                logger.error("[Runner] CriticAgent 输出解析失败，跳过校验环节。")
                break
            
            current_round += 1
        logger.info("[Runner] 协作流程结束，生成最终报告。")

        '''
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
        
        '''
        resp = self.llm.chat([
            Message(role="system", content="你是舆情分析助手。"),
            Message(role="user", content=f"给我一句话概括：{topic}"),
        ])
        '''
        # 组装最终的 ReportIR

        # 提取来源连接
        # extracted_sources = _extract_sources(final_raw_evidence)

        '''
        # 从final_insight_data中直接获取大模型提取的sources
        sources = final_insight_data.get("sources",[])
        summary_text = final_insight_data.get('summary','暂无摘要数据')
        sentiment_text = final_insight_data.get('sentiment','未知')
        key_points = final_insight_data.get('key_points', [])
        '''

        '''
        # 注意：这里展示了如何把 Insight Agent 提炼的 key_points 传给 schema
        ir = ReportIR(
            topic=topic, 
            # summary=query_result,   # 暂时用queryagent汇报顶替
            # summary=f"【情感倾向：{insight_data.get('sentiment', '未知')}】 {insight_data.get('summary', '')}",
            summary=f"【情感倾向：{sentiment_text}】 {summary_text}",
            # key_points=insight_data.get("key_points", []),
            key_points=final_insight_data.get("key_points", []),
            sources=sources            # 暂时留空
            )
        '''

        # 加入state后，组装IR
        # 摘要和要点来自 InsightAgent 的大脑
        # 来源 URL 来自底层工具偷偷写在黑板上的 state.sources
        final_data_str = json.dumps({
            "topic": topic,
            "sentiment": final_insight_data.get('sentiment','未知'),
            "summary": final_insight_data.get('summary','暂无摘要数据'),
            "key_points": final_insight_data.get('key_points', []),
            "sources": state.sources # 直接从黑板上拿，不会出现幻觉
        },ensure_ascii=False, indent=2)

        '''
        # 接入ReportAgent后，把所有信息打包成一个字符串，喂给主编
        final_data_str = json.dumps({
            "topic": topic,
            "sentiment": sentiment_text,
            "summary": summary_text,
            "key_points": key_points,
            "sources": sources
        },ensure_ascii=False, indent=2)
        '''

        write_jsonl(self.trace_path, {"ts": now_iso(), "event": "run_end", "topic": topic})
        
        # return ir
        # 接入ReportAgent后，将生成器的控制权交给ReprotAgent
        return self.report_agent.generate_stream(final_data_str)