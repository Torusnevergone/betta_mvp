# app/tools/vision_tool.py
from __future__ import annotations
import json
from typing import Optional
from app.tools.base import BaseTool
from app.core.llm_client import LLMClient
from app.core.state import SessionState
from app.core.logging import setup_logger
from app.core.types import Message
logger = setup_logger()
class VisionAnalysisTool(BaseTool):
    """
    视觉分析工具。
    专门用于调用多模态模型（如 Qwen-VL）来理解和描述网络图片的内容。
    """
    name = "analyze_image"
    description = "当你需要理解一张网络图片的内容时，必须调用此工具。请传入图片的公共URL链接，它会返回对图片画面的详细文字描述。"
    parameters = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "需要分析的图片的公共 URL 链接（通常以 http/https 开头，以 jpg/png/webp 结尾）。"
            }
        },
        "required": ["image_url"]
    }
    def __init__(self, llm_client: LLMClient):
        """
        依赖注入：在实例化这个 Tool 时，必须把主流程里的 llm 实例传进来。
        """
        # 我们需要传入全局的 llm_client
        self.llm_client = llm_client    # 把传进来的实例保存为自己的属性
        super().__init__()
    def run(self, state: Optional[SessionState] = None, **kwargs) -> str:
        image_url = kwargs.get("image_url", "")
        if not image_url:
            return json.dumps({"error": "图片 URL 不能为空"})
        logger.info(f"[Vision Tool] 正在请求视觉模型分析图片: {image_url}")
        # 构造通义千问-VL / GPT-4o 兼容的视觉消息格式
        # 注意：这里的 content 是一个列表，包含 text 和 image_url
        # 将python原生字典改为Message对象
        messages = [Message(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "你是一个专业的舆情图像分析师。请详细描述这张图片的内容。如果图片中包含文字（如截图、公告）、车辆损坏情况（如碰撞、起火）、或者人物情绪，请务必详细提取并描述。不要进行主观猜测，只描述客观事实。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            )]
        try:
            # 核心：调用 LLMClient，并且务必设置 use_vision=True ！
            # 核心：使用自己身上保存的 llm_client 实例去发起请求，并指定 use_vision=True
            response = self.llm_client.chat(
                messages=messages,
                use_vision=True  # 这会让请求路由到 Qwen-VL
            )
            
            description = response.content
            logger.info(f"[Vision Tool] 模型返回图片描述: {description[:50]}...")
            
            return (
                f"【图片视觉分析报告】\n"
                f"输入图片URL: {image_url}\n"
                f"画面内容描述: {description}"
            )
        except Exception as e:
            logger.error(f"[Vision Tool] 调用视觉 API 失败: {str(e)}")
            return f"图片分析失败，原因: {str(e)}。无法获取该图片的视觉信息，请仅依赖文本线索进行分析。"