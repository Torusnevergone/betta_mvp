# app/main.py
from __future__ import annotations
import os
import sys
from pathlib import Path
# 必须在最顶层设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from flask import Flask, request, Response, render_template
from app.core.logging import setup_logger
from app.core.llm_client import LLMClient
from app.orchestrator.runner import Runner
from app.orchestrator.forum_engine import ForumEngine

from flask import stream_with_context

from app.storage.db_manager import init_db, save_message, get_recent_history

import json
from flask import request, jsonify
from app.services.sentiment_service import sentiment_model # 导入单例模型服务

logger = setup_logger()
# 1. 初始化 Flask 应用
# 因为 main.py 在 app 目录下，而 templates 在根目录，
# 所以我们需要动态计算模板文件夹的绝对路径。
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
# 启动应用前
init_db()
# 2. 页面路由：返回前端页面
@app.route("/", methods=["GET"])
def read_root():
    return render_template("index.html")

# 加入独立部署的LoRA情感分析API路由
@app.route("/api/sentiment", methods=["POST"])
def predict_sentiment_api():
    """
    API 网关层。只负责接收参数和返回 JSON，不包含任何计算逻辑。
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "缺少 text 参数"}), 400
    text = data["text"]
    # 核心：调用Service层的业务逻辑
    try:
        result = sentiment_model.predict(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[API Gateway] 模型推理失败: {e}")
        return jsonify({"error": "内部模型服务异常"}), 500

# 3. 核心 SSE 接口路由 (Flask 版本)
@app.route("/api/analyze", methods=["GET"])
def analyze_topic():
    # Flask 获取 URL 参数的方式
    topic = request.args.get("topic", "")
    # 引入db
    session_id = request.args.get("session_id", "default_session")
    
    if not topic:
        return {"error": "话题不能为空"}, 400
    logger.info(f"[Flask] 收到前端分析请求，话题: {topic}")

    # 架构升级：加入db 从数据库中捞取滑动窗口历史记录!
    user_history = get_recent_history(session_id, limit=3)
    if user_history:
        logger.info(f"[Memory] 成功从数据库加载 {len(user_history)} 条历史记录。")

    # 为了保证并发隔离，每次请求实例化独立的客户端和 Runner
    llm = LLMClient()
    
    # 改为论坛机制
    # runner = Runner(llm=llm)
    runner = ForumEngine(llm=llm)

    # 这是一个生成器函数，用于把 Runner 的输出包装成 SSE 格式
    # 修改为异步生成
    def generate_events():  # 去掉 async
        full_report_content = ""
        try:
            # 恢复普通的 for 循环
            # for chunk in runner.run(topic):
            # 把历史记录传给ForumEngine
            for chunk in runner.run(topic, history=user_history):
                full_report_content += chunk
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
            
            yield "data: [DONE]\n\n"

            # 对话结束后，把这一轮的问答持久化到数据库中！
            save_message(session_id, "user", topic)
            save_message(session_id, "assistant", full_report_content)
            logger.info(f"[Memory] 本轮对话已持久化至数据库。")
            
            from app.storage.io import save_text
            import os
            os.makedirs("app/storage", exist_ok=True)
            save_text("app/storage/report.md", full_report_content)
            logger.info(f"[Flask] 报告已成功落盘保存到 app/storage/report.md")
            
        except Exception as e:
            logger.error(f"生成流时发生错误: {e}")
            yield f"data: [ERROR] 服务器内部错误: {str(e)}\n\n"
    # Flask 实现 SSE 的核心：Response 返回生成器，并指定 mimetype='text/event-stream'
    # return Response(generate_events(), mimetype="text/event-stream")
    # 使用 stream_with_context 包装生成器，防止上下文丢失
    return Response(stream_with_context(generate_events()), mimetype="text/event-stream")
if __name__ == "__main__":
    # 启动 Flask 开发服务器
    logger.info("启动 Flask Web 服务器，请在浏览器访问 http://127.0.0.1:5000")
    # threaded=True 允许 Flask 在开发模式下处理多个并发请求（比如一个页面的加载和一个 SSE 连接）
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)