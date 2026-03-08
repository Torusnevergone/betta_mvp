from __future__ import annotations
import json
from typing import Generator, List
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.core.types import Message
from app.core.logging import setup_logger
import networkx as nx
logger = setup_logger()

class ReportAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        # 注意：这里的 role_prompt 变简短了，因为具体的要求会在 Prompt 链的各个环节中动态生成
        super().__init__(
            name="ReportAgent",
            role_prompt="你是顶级的商业舆情报告主编，擅长撰写逻辑严密、排版精美、图文并茂的分析报告。",
            llm=llm,
            tools=None
        )

    def _extract_knowledge_graph(self, forum_logs:List[str]) -> nx.Graph:
        """
        利用 LLM 将非结构化的论坛聊天记录，抽取为结构化的三元组，并构建内存知识图谱。
        """
        logger.info("[GraphRAG] 正在从论坛日志中抽取知识图谱实体与关系...")

        # 1. 构造抽取 Prompt
        logs_text = "\n".join(forum_logs)
        extract_prompt = (
            f"请从以下会议记录中，提取出最核心的事实关系（实体-关系-实体）。\n"
            f"会议记录：\n{logs_text}\n\n"
            f"【强制要求】：只输出 JSON 格式的列表，不要有任何其他文字。格式如：\n"
            f'[["小米SU7", "存在问题", "车机卡死"], ["车机卡死", "客诉数量", "2起"]]\n'
        )
        # 2.调用LLM进行信息抽取
        response_str = self.llm.chat([Message(role="user", content=extract_prompt)]).content.strip()
        # 3.解析清洗
        response_str = response_str.replace("```json","").replace("```","").strip()
        triplets = []
        try:
            triplets = json.loads(response_str)
        except json.JSONDecodeError:
            logger.warning("[GraphRAG] 三元组抽取失败，降级为无图谱模式。")
            return nx.Graph()
        # 4.构建NetworkX内存图谱
        G = nx.Graph()
        for triplet in triplets:
            if isinstance(triplet, list) and len(triplet) == 3:
                head, rel, tail = triplet
                # 添加节点和边
                G.add_edge(head, tail, relation=rel)
        logger.info(f"[GraphRAG] 成功构建局部知识图谱，包含 {G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边。")
        return G
        
    def _graph_to_text(self, G:nx.graph()) -> str:
        """将图谱转化为人类/LLM可读的文本，作为 Hard Prompt 注入"""
        if G.number_of_edges() == 0:
            return "暂无图谱事实"
        facts = []
        for u, v, data in G.edges(data=True):
            facts.append(f"- {u} --[{data['relation']}]--> {v}")
        return "\n".join(facts)
    
    # 核心：添加多轮Prompt-Chain (模板选择 -> 篇幅规划 -> 章节生成）实现报告自动化排版。
    def generate_stream(self, structured_data: str, forum_logs: List[str] = None) -> Generator[str, None, None]:
        # 执行GraphRAG抽取（前置动作）
        G = self._extract_knowledge_graph(forum_logs or [])
        graph_facts = self._graph_to_text(G)
        # 解析一下传入的结构化数据，方便后面用
        try:
            data_dict = json.loads(structured_data)
            topic = data_dict.get("topic", "未知话题")
            sentiment = data_dict.get("sentiment", "未知")
        except:
            topic = "未知话题"
            sentiment = "未知"

        # 链条环节 1：模板选择 (Template Selection)
        logger.info(f"[{self.name}] 链条环节 1：正在进行模板选择...")
        template_prompt = (
            f"目标话题是：【{topic}】，当前整体情感倾向为：【{sentiment}】。\n"
            f"请判断应该使用以下哪种报告模板：\n"
            f"A. 危机公关应对模板 (适合负面、事故、争议类话题)\n"
            f"B. 产品口碑评测模板 (适合试驾、发布会、评测类话题)\n"
            f"C. 行业趋势分析模板 (适合宏观、竞品对比、技术类话题)\n"
            f"只需输出 A、B 或 C 即可，不要其他废话。"
        )
        template_choice = self.llm.chat([Message(role="user", content=template_prompt)]).content.strip()
        template_name = "产品口碑评测模板"
        if "A" in template_choice: template_name = "危机公关应对模板"
        elif "C" in template_choice: template_name = "行业趋势分析模板"
        logger.info(f"[{self.name}] 决定采用：{template_name}")

        # 链条环节 2：篇幅规划 (Outline Generation)
        logger.info(f"[{self.name}] 链条环节 2：正在生成报告大纲...")
        outline_prompt = (
            f"请基于选定的【{template_name}】，为话题【{topic}】生成一份包含 4 个核心章节的报告大纲。\n"
            f"大纲必须逻辑严密，层层递进。直接输出大纲文本，不需要写具体内容。"
        )
        outline = self.llm.chat([Message(role="user", content=outline_prompt)]).content.strip()
        logger.info(f"[{self.name}] 大纲生成完毕。")

        # 链条环节 3：章节生成与动态图表注入 (Chapter Generation & ECharts)
        logger.info(f"[{self.name}] 链条环节 3：正在流式撰写正文并注入图表代码...")
        final_prompt = (
            f"请作为主编，基于以下【报告大纲】撰写最终的 Markdown 舆情报告：\n{outline}\n\n"
            f"【写作素材与依据】：\n"
            f"1. 核心事实图谱（防幻觉依据）：\n{graph_facts}\n"
            f"2. 分析师提交的数据：\n{structured_data}\n\n"
            f"【高阶排版与图表注入要求】（极其重要）：\n"
            f"1. 在报告的“情感倾向”或“数据统计”章节，你必须插入一段 ECharts 的 JSON 配置代码，用于前端渲染可视化图表（比如饼图或柱状图）。\n"
            f"2. 图表代码必须用 ``` echarts 和 ``` 包裹起来。例如：\n"
            f"```echarts\n"
            f'{{"title": {{"text": "情感分布"}}, "series": [{{"type": "pie", "data": [{{"value": 80, "name": "负面"}}, {{"value": 20, "name": "正面"}}]}}]}}\n'
            f"```\n"
            f"3. 报告末尾无需你来写参考来源列表，系统会自动拼接。\n"
            f"请开始流式撰写全文："
        )
        messages = [
            Message(role="system", content=self.role_prompt),
            Message(role="user", content=final_prompt)
        ]
        # 只有最后一步才是流式输出给前端的！
        for chunk in self.llm.chat_stream(messages=messages, tools=None):
            if chunk is not None and str(chunk) != "":
                yield str(chunk)