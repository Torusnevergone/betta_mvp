import random
from app.core.logging import setup_logger
logger = setup_logger()

class SentimentModelService:
    """
    模拟部署在 GPU 服务器上的 LoRA 微调模型服务。
    在真实环境中，这里会包含 transformers 库的模型加载和 Inference 逻辑。
    """
    def __init__(self):
        self.model_version = "qwen-1.5b-sentiment-lora-v2"
        # 模拟模型加载耗时
        logger.info(f"[Model Service] 正在加载微调模型权重 {self.model_version} 到显存...")

        # 垂直领域词典
        self.negative_words = ["卡死", "死机", "异响", "起火", "退订", "失望", "退货", "召回", "事故", "瑕疵", "局促", "胎噪"]
        self.positive_words = ["流畅", "惊艳", "遥遥领先", "丝滑", "好评", "高级", "质感", "火爆", "认可", "扎实", "稳健"]

    def predict(self, text:str) -> dict:
        """执行模型推理，返回极性和置信度"""
        logger.info(f"[Model Service] 收到推理请求，文本长度: {len(text)}")

        score = 0.5
        for w in self.negative_words:
            if w in text: score -= 0.15
        for w in self.positive_words:
            if w in text: score += 0.15
        score = max(0.01, min(0.99, score+random.uniform(-0.02, 0.02)))

        if score > 0.6:
            sentiment = "正面"
            confidence = score
        elif score < 0.4:
            sentiment = "负面"
            confidence = 1.0-score
        else:
            sentiment = "中立"
            confidence = 0.5+abs(score-0.5)
        
        return {
            "model_version": self.model_version,
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "core_emotion_score": round(score, 4)
        }

# 单例模式：保证全局只有一个模型实例（避免把显存撑爆）
sentiment_model = SentimentModelService()