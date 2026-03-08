# app/main.py
from __future__ import annotations
import os

# 必须在最顶层设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.logging import setup_logger
from app.core.llm_client import LLMClient
from app.orchestrator.runner import Runner

logger = setup_logger()

# 1. 初始化 FastAPI 应用
app = FastAPI(title="BettaFish MVP", description="多 Agent 舆情分析系统")

# 2. 配置模板引擎（指向我们刚才建的 templates 文件夹）
templates = Jinja2Templates(directory="templates")

# 3. 页面路由：当用户访问根目录 "/" 时，返回 HTML 网页
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 4. 核心 SSE 接口路由
@app.get("/api/analyze")
async def analyze_topic(topic: str):
    """
    接收前端传来的 topic，启动 Runner，并以 SSE 流的形式返回数据。
    """
    logger.info(f"收到前端分析请求，话题: {topic}")
    
    # 每次请求都实例化一个新的 LLMClient 和 Runner，保证并发时的状态隔离
    llm = LLMClient()
    runner = Runner(llm=llm)

    # 这是一个异步的生成器函数，专门用来把 Runner 吐出的文字包装成 SSE 格式
    async def event_generator():
        try:
            # runner.run() 是一个同步的生成器（它里面有 while 循环和 time.sleep 等）
            # 在真实的生产环境中，这里应该用线程池或者把 Runner 彻底改成 async。
            # 但在 MVP 阶段，FastAPI 允许我们在 async 函数里直接 yield 同步生成器的数据。
            stream = runner.run(topic)
            
            for chunk in stream:
                # SSE 协议规范：必须以 "data: 内容\n\n" 的格式发送
                # 因为 chunk 里可能自带换行符（\n），直接发会破坏 SSE 协议，
                # 所以我们把换行符替换成一个特殊的占位符 "\\n"，前端收到后再换回来。
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
            
            # 报告生成完毕，发送一个结束标记给前端
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"生成流时发生错误: {e}")
            yield f"data: [ERROR] 服务器内部错误: {str(e)}\n\n"

    # 使用 FastAPI 提供的 StreamingResponse，并指定 media_type 为 text/event-stream
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# 5. 启动服务器
if __name__ == "__main__":
    # 使用 uvicorn 启动服务器，监听 0.0.0.0 的 8000 端口
    logger.info("启动 Web 服务器，请在浏览器访问 http://127.0.0.1:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)