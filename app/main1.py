from __future__ import annotations
import sys
import os

# 在导入任何 AI 相关的库之前，设置 HuggingFace 国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from app.core.logging import setup_logger
from app.core.llm_client import LLMClient
from app.orchestrator.runner import Runner
from app.report.render import render_markdown
from app.storage.io import save_text

def main(argv: list[str]) -> int:
    logger = setup_logger()
    if len(argv) < 2:
        logger.error('用法：python -m app.main "你的话题"')
        return 2

    topic = argv[1].strip()
    llm = LLMClient()
    runner = Runner(llm=llm)
    logger.info(f"Running topic: {topic}")

    # ir = runner.run(topic)
    # md = render_markdown(ir)
    # runner现在返回的是一个生成器
    stream = runner.run(topic)
    print("\n\n" + "="*50)
    print("最终舆情报告 (实时生成中)")
    print("\n\n" + "="*50)
    full_markdown = ""
    # 实时流式输出
    for chunk in stream:
        # end="" 保证不换行，flush=True 保证立即刷新到屏幕上
        print(chunk, end="", flush=True)
        full_markdown += chunk
    
    print("\n\n" + "="*50)

    out_path = "app/storage/report.md"
    # save_text(out_path, md)
    save_text(out_path, full_markdown)
    logger.info(f"Report saved: {out_path}")
    # print(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))