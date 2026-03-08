from __future__ import annotations
import json
from typing import Generator, List, Tuple
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.core.types import Message
from app.core.logging import setup_logger
import networkx as nx

logger = setup_logger()

class ReportAgent(BaseAgent):
    def __init__(self,llm:LLMClient):
        role_prompt = (
            "你是专业的舆情报告主编 (Report Agent)。\n"
            "你的任务是：接收前面分析师整理好的数据，并将其扩写、排版成一篇专业的 Markdown 舆情报告。\n"
            "【强制要求】：\n"
            "1. 报告必须包含：标题、执行摘要、情感倾向、关键发现、参考来源列表。\n"
            "2. 在【参考来源列表】中，必须严格按照 '[序号]. [标题](URL)' 的 Markdown 链接格式输出。\n"
            "3. 严禁篡改数据，必须严格基于输入的事实图谱和结构化数据进行扩写。\n"
            "4. 直接输出 Markdown 文本，不要用 ```markdown 包裹。"
        )
        super().__init__(
            name="ReportAgent",
            role_prompt=role_prompt,
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
    
    def generate_stream(self, structured_data:str, forum_logs:List[str] = None) -> Generator[str, None, None]:
        """
        流式生成报告（融合 GraphRAG 校验）
        """
        # 1.执行GraphRAG抽取
        G = self._extract_knowledge_graph(forum_logs or [])
        graph_facts = self._graph_to_text(G)
        # 2.融合图谱事实与结构化数据
        final_prompt = (
            f"【GraphRAG 核心事实校验图谱】（请严格基于此图谱撰写关键发现，防幻觉）：\n"
            f"{graph_facts}\n\n"
            f"【分析师提交的结构化数据与来源】：\n"
            f"{structured_data}\n\n"
            f"请基于以上两部分材料，生成最终的舆情报告。"
        )
        messages = [
            Message(role="system", content=self.role_prompt),
            Message(role="user", content=final_prompt)
        ]
        logger.info(f"[{self.name}] 正在基于 GraphRAG 事实图谱奋笔疾书...\n")

        for chunk in self.llm.chat_stream(messages=messages, tools=None):
            if chunk is not None and str(chunk) != "":
                yield str(chunk)