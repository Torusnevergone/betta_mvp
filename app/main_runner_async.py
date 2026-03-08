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

from flask import stream_with_context

logger = setup_logger()
# 1. 初始化 Flask 应用
# 因为 main.py 在 app 目录下，而 templates 在根目录，
# 所以我们需要动态计算模板文件夹的绝对路径。
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
# 2. 页面路由：返回前端页面
@app.route("/", methods=["GET"])
def read_root():
    return render_template("index.html")
# 3. 核心 SSE 接口路由 (Flask 版本)
@app.route("/api/analyze", methods=["GET"])
def analyze_topic():
    # Flask 获取 URL 参数的方式
    topic = request.args.get("topic", "")
    
    if not topic:
        return {"error": "话题不能为空"}, 400
    logger.info(f"[Flask] 收到前端分析请求，话题: {topic}")
    # 为了保证并发隔离，每次请求实例化独立的客户端和 Runner
    llm = LLMClient()
    runner = Runner(llm=llm)
    # 这是一个生成器函数，用于把 Runner 的输出包装成 SSE 格式
    # 修改为异步生成
    def generate_events():  # 去掉 async
        full_report_content = ""
        try:
            # 恢复普通的 for 循环
            for chunk in runner.run(topic):
                full_report_content += chunk
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
            
            yield "data: [DONE]\n\n"
            
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