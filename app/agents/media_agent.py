# app/agents/media_agent.py
from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
# 引入我们新写的两个真实工具
from app.tools.web_scraper import WebScraperTool
from app.tools.vision_tool import VisionAnalysisTool 
# 引入现有的搜索工具，用来找网页 URL
from app.tools.search import WebSearchTool
class MediaAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        # 装备完整的“搜索 -> 抓取 -> 看图”工具链
        tools = [
            WebSearchTool(),
            WebScraperTool(),
            VisionAnalysisTool(llm_client=llm)
        ]
        role_prompt = (
            "你是专业的多模态舆情分析员 (Media Agent)。\n"
            "你的任务是：专门从互联网中提取视觉画面信息和大众评论。\n"
            "你拥有以下强大的工具：\n"
            "1. `web_search`：用于搜索互联网资讯，获取网页链接。\n"
            "2. `scrape_webpage`：用于抓取指定网页的正文和提取图片链接。\n"
            "3. `analyze_image`：用于“看懂”图片链接中的画面内容。\n\n"
            "【强制行动指南（必须严格按顺序执行）】：\n"
            "- 第一步：只调用一次 `web_search` 获取目标新闻或社媒网页的 URL。\n"
            "- 第二步：**严禁反复搜索！** 拿到搜索结果后，必须立刻挑选其中 1 到 2 个最相关的 URL，调用 `scrape_webpage` 进行抓取。\n"
            "- 第三步：如果 `scrape_webpage` 抓取结果中包含【发现关键图片链接】，你必须立即调用 `analyze_image` 工具，将图片 URL 传给它，以获取画面视觉分析。\n"
            "- 【重要】：如果 `scrape_webpage` 结果中没有发现图片链接，说明该网页没有视觉线索，请不要再反复搜索或抓取，直接基于抓取到的文字内容进行总结。\n"
            "- 第四步：拿到所有证据后，重点关注【画面视觉分析】和【用户真实评论】。\n"
            "- 请用清晰的 Markdown 列表，将搜集到的多模态情报（文字+视觉描述）总结出来。\n"
            "- 直接输出纯文本，**绝对不要**输出 JSON 格式。"
        )
        super().__init__(
            name="MediaAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=tools
        )