from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from duckduckgo_search import DDGS
from ddgs import DDGS
from app.core.types import Evidence
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState

logger = setup_logger()

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "当需要获取最新的互联网资讯、新闻、或者补充事实证据时，调用此工具。输入一个搜索关键词，返回相关的网页标题、链接和摘要。"

    # 这里的 parameters 必须严格遵守 JSON Schema 规范，大模型就是看这个说明书来决定怎么传参的
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要搜索的关键词，例如 '小米SU7 最新评测' 或者 'SpaceX 2026 发射计划'",
            },
            "max_results": {
                "type": "integer",
                # "description": "期望返回的最大结果数量，默认为 5",
                "default": 5
            }
        },
        "required": ["query"]
    }

    def run(self, state:Optional[SessionState] = None, **kwarg) -> str:
        """
        实现 BaseTool 要求的 run 方法。
        注意：Agent 框架中，Tool 的返回值通常要求是字符串（JSON 字符串或纯文本格式），
        以便直接塞进 LLM 的上下文里。
        """
        query = kwarg.get("query")
        max_results = kwarg.get("max_results", 5)

        if not query:
            return json.dumps({"error": "搜索词(query)不能为空"})
        logger.info(f"[Tool Call] 正在执行 web_search, 关键词: 【{query}】")

        # 加入state之后的修改，该列表专门给LLM砍，没有URL
        llm_readable_results = []

        # results_evidence: List[Dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                '''
                for r in results:
                    # 先把它转成字典，方便最后 dumps 成 JSON 字符串
                    ev = {
                        "source": "Web (DuckDuckgo)",
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    results_evidence.append(ev)
                '''
                for i, r in enumerate(results):
                    title = r.get("title", "")
                    snippet = r.get("body", "")
                    url = r.get("href", "")

                    # 1.组装给LLM砍的文本，只给它看标题和摘要，省token
                    # llm_readable_results.append(f"[结果{i+1}]标题：{title}\n摘要：{snippet}")
                    # 加入真实media_agent后，需要把url传给llm，从而传给爬虫工具
                    llm_readable_results.append(f"[结果{i+1}]\n标题：{title}\n链接：{url}\n摘要：{snippet}")
                    # 2.行业规范：旁路存储 旁路写入state
                    if state:
                        state.add_source(title=title, url=url)

        except Exception as e:
            return json.dumps({"error": f"搜索失败: {str(e)}"})

        # 将结果转成 JSON 字符串返回给大模型
        # return json.dumps(results_evidence, ensure_ascii=False)

        # 加入旁路存储后
        return "\n\n".join(llm_readable_results)