from __future__ import annotations
import json
import httpx
from typing import Optional
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState
logger = setup_logger()

class SentimentAnalysisTool(BaseTool):
    """
    专门调用内部微调模型 API 的工具。
    取代了大模型自带的模糊情感判断，提供带有置信度的精准情感分类。
    """
    name = "predict_sentiment"
    description = "当需要对长篇舆情证据进行精准的情感倾向判定时，必须调用此工具。它会连接到内部微调过的专属情感模型，返回极性与置信度。"
    parameters = {
        "type": "object",
        "properties": {
            "text_summary": {
                "type": "string",
                "description": "需要分析的舆情文本摘要（请将长篇证据压缩为不超过500字的摘要后传入）"
            }
        },
        "required": ["text_summary"]
    }

    def run(self, state:Optional[SessionState] = None, **kwargs) -> str:
        text_summary = kwargs.get("text_summary", "")
        if not text_summary:
            return json.dumps({"error": "待分析文本不能为空"})

        logger.info(f"[Tool Call] 正在请求内部微调模型 API (LoRA)...")
        # 这里的 127.0.0.1:5000 就是我们刚才在 Flask 里写的那个路由！
        api_url = "http://127.0.0.1:5000/api/sentiment"

        try:
            # 发起 HTTP POST 请求，模拟微服务间的 RPC 调用
            response = httpx.post(api_url, json={"text": text_summary}, timeout=5.0)
            response.raise_for_status()
            result_data = response.json()

            logger.info(f"[Sentiment API] 模型返回结果: {result_data['sentiment']} (置信度: {result_data['confidence']})")

            # 格式化输出给agent看
            output = (
                f"【专属微调模型预测结果】\n"
                f"模型版本: {result_data['model_version']}\n"
                f"情感倾向: {result_data['sentiment']}\n"
                f"预测置信度: {result_data['confidence'] * 100}%\n"
                f"请在最终报告中引用此倾向和置信度。"
            )
            return output

        except Exception as e:
            logger.error(f"[Sentiment API] 调用失败: {str(e)}")
            return f"情感分析 API 调用失败: {str(e)}。请自行根据文本判断情感倾向。"