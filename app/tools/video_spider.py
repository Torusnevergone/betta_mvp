from __future__ import annotations
import json
import time
from typing import Optional
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState

logger = setup_logger()

class VideoSpiderTool(BaseTool):
    """
    短视频/多模态平台抓取工具。
    模拟从抖音、B站、小红书等平台提取视频画面描述和高赞评论。
    """
    name = "video_search"
    description = "当需要获取短视频平台（如抖音、B站、小红书）上的视觉画面信息、用户开箱体验或高赞评论时，调用此工具。"
    parameters = {
        "type": "object",
        "properties":{
            "query":{
                "type":"string",
                "description":"要搜索的短视频关键词，例如 '小米SU7 试驾 抖音'",
            }
        },
        "required":["query"]
    }

    def run(self, state:Optional[SessionState] = None, **kwargs) -> str:
        query = kwargs.get("query", "")
        if not query:
            return json.dumps({"error": "搜索词(query)不能为空"})

        logger.info(f"[Tool Call] 正在执行 video_search (多模态模拟), 关键词: 【{query}】")

        # 模拟多模态模型的返回结果（包含画面视觉描述和评论区文本）
        mock_results = [
            {
                "platform": "抖音",
                "title": f"关于 {query} 的热门视频",
                "visual_description": "[视觉模型提取] 画面中人群密集围观展车，多人拿手机拍摄。展车颜色为海湾蓝，漆面反光度高。博主表情激动，竖起大拇指。",
                "top_comments": "1. 卧槽这漆面质感绝了！(15w赞)\n2. 空间看起来有点小啊，后排能坐1米8的吗？(8w赞)",
                "url": "https://douyin.com/mock_video_1"
            },
            {
                "platform": "B站",
                "title": f"{query} 深度评测",
                "visual_description": "[视觉模型提取] 画面展示了车机屏幕的滑动操作，无卡顿掉帧。后续画面为底盘升降特写，展示空气悬挂。",
                "top_comments": "1. 课代表来了：底盘扎实，车机丝滑，但胎噪略大。(2.1w赞)\n2. 纯路人，这套UI设计确实遥遥领先。(1.5w赞)",
                "url": "https://bilibili.com/mock_video_2"
            }
        ]
        llm_readable_results = []
        for i, r in enumerate(mock_results):
            # 组装给大模型看的纯文本，重点突出视觉描述
            content = (
                f"[视频 {i+1} | 平台: {r['platform']}]\n"
                f"标题：{r['title']}\n"
                f"画面视觉分析：{r['visual_description']}\n"
                f"评论区高赞：\n{r['top_comments']}"
            )
            llm_readable_results.append(content)
            # 行业规范：将视频 URL 旁路写入黑板
            if state:
                state.add_source(title=f"[{r['platform']}] {r['title']}", url=r['url'])
        return "\n\n".join(llm_readable_results)