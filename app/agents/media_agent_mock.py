# app/agents/media_agent.py
from __future__ import annotations
from app.agents.base import BaseAgent
from app.core.llm_client import LLMClient
from app.tools.video_spider import VideoSpiderTool
class MediaAgent(BaseAgent):
    def __init__(self, llm: LLMClient):
        # 定义 Media Agent 的人设 prompt
        role_prompt = (
            "你是专业的多模态舆情分析员 (Media Agent)。\n"
            "你的任务是：专门从短视频平台（抖音、B站、小红书）提取视觉信息和大众评论。\n"
            "你拥有 video_search 工具，它可以“看懂”视频画面并提取高赞评论。\n"
            "【行动指南】：\n"
            "- 拿到证据后，重点关注【画面视觉分析】和【用户真实评论】。\n"
            "- 请用清晰的 Markdown 列表，将搜集到的多模态情报总结出来。\n"
            "- 直接输出纯文本，**绝对不要**输出 JSON 格式。"
        )
        
        # 实例化它专属的多模态工具
        tools = [VideoSpiderTool()]
        super().__init__(
            name="MediaAgent",
            role_prompt=role_prompt,
            llm=llm,
            tools=tools
        )