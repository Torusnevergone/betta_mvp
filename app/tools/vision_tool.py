# app/tools/vision_tool.py
from __future__ import annotations
import json
from typing import Optional
from app.tools.base import BaseTool
from app.core.llm_client import LLMClient
from app.core.state import SessionState
from app.core.logging import setup_logger
from app.core.types import Message
import requests
import base64
logger = setup_logger()
# 【新增函数】：负责下载图片并转码，这样qwenvl就不用再去访问这个图片url了
def get_image_base64(url: str) -> Optional[str]:
    """下载图片并转换为 base64 编码字符串"""
    try:
        # 伪装成真实的浏览器，绕过防盗链
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # 提取域名作为 Referer，对付更严格的防盗链
            "Referer": "/".join(url.split("/")[:3]) + "/" 
        }
        logger.info(f"[Vision Tool] 正在尝试下载图片转码: {url}")
        
        # 设置 10 秒超时
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 获取 Content-Type，默认为 image/jpeg
        content_type = response.headers.get('Content-Type', 'image/jpeg')
        if not content_type.startswith('image/'):
            content_type = 'image/jpeg'
            
        # 转换为 base64
        b64_data = base64.b64encode(response.content).decode('utf-8')
        
        # 拼接成 standard data URI 格式，这是大模型 API 支持的格式
        return f"data:{content_type};base64,{b64_data}"
    except Exception as e:
        logger.error(f"[Vision Tool] 下载图片失败 {url}: {e}")
        return None

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
        # 1. 先把图片下载下来转成 Base64
        base64_image = get_image_base64(image_url)
        # 如果下载失败，直接返回错误信息给 Agent，不请求大模型了
        if not base64_image:
            return f"图片分析失败：无法从目标网站下载图片（可能存在防盗链或链接已失效）。请仅依赖网页的文字内容进行分析。"
        # 2. 构造发送给大模型的 JSON，使用 Base64 字符串代替 URL
        messages = [Message(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "你是一个专业的舆情图像分析师。请详细描述这张图片的内容。如果图片中包含文字（如截图、公告）、车辆损坏情况（如碰撞、起火）、或者人物情绪，请务必详细提取并描述。不要进行主观猜测，只描述客观事实。"
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