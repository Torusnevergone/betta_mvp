# app/tools/web_scraper.py
from __future__ import annotations
import json
from typing import Optional
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState
logger = setup_logger()
class WebScraperTool(BaseTool):
    """
    基于 Playwright 的真实网页抓取工具。
    能够渲染动态网页，提取核心文本以及高质量的图片链接。
    """
    name = "scrape_webpage"
    description = "当你需要获取某个具体网页的详细正文内容和图片资源时，调用此工具。请传入一个有效的网页URL。"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的网页的完整 URL (必须以 http:// 或 https:// 开头)"
            }
        },
        "required": ["url"]
    }
    def run(self, state: Optional[SessionState] = None, **kwargs) -> str:
        url = kwargs.get("url", "")
        if not url.startswith("http"):
            return json.dumps({"error": "无效的 URL。"})
        logger.info(f"[Web Scraper] 正在启动无头浏览器抓取: {url}")
        try:
            with sync_playwright() as p:
                # 启动 Chromium，headless=True 表示不弹出真实浏览器窗口
                # 尝试修正gpu参数，不启用gpu
                # browser = p.chromium.launch(headless=True)
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',               # <--- 加上这个
                        '--disable-software-rasterizer' # <--- 加上这个
                    ]
                )
                # 伪装成正常的手机/电脑浏览器，防止被简单的反爬拦截
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                page = context.new_page()
                
                try:
                    # 设置超时时间为 15 秒，等待网络空闲
                    # page.goto(url, timeout=15000, wait_until="networkidle")
                    # 【优化1】：将超时时间延长到 20 秒，并且只要 DOM 加载完就算成功
                    page.goto(url, timeout=20000, wait_until="domcontentloaded")
                    # 【优化2】：稍微等 1 秒，让一些简单的 JS 渲染一下内容
                    page.wait_for_timeout(1000)
                
                    # 获取渲染后的完整 HTML
                    html_content = page.content()
                except Exception as e:
                    return f"抓取失败: 目标网页存在反爬或加载超时 ({str(e)})"

                    browser.close()
            # 使用 BeautifulSoup 解析 HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            # 1. 提取网页标题
            title = soup.title.string if soup.title else "未知标题"
            # 2. 提取正文文本 (移除脚本和样式)
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            text_content = soup.get_text(separator=' ', strip=True)
            # 截断文本，防止把 LLM 的上下文窗口撑爆
            text_content = text_content[:1500] + ("..." if len(text_content) > 1500 else "")
            # 3. 提取图片链接 (寻找 <img> 标签)
            image_urls = []
            for img in soup.find_all('img'):
                # src = img.get('src')
                # 加入修复：防止抓不到图片，检查多种可能的真实图片地址属性 (适配懒加载)
                src = img.get('src') or img.get('data-src') or img.get('data-original') or img.get('data-lazy-src')
                # 过滤掉头像、小图标，尽量只保留大图，这里加了一个base64
                if src and src.startswith("http") and not any(x in src for x in ['avatar', 'icon', 'logo', 'gif', 'base64']):
                    image_urls.append(src)
            
            # 只取前 2 张最具代表性的图片，避免调用视觉API太贵/太慢
            top_images = image_urls[:2]
            logger.info(f"[Web Scraper] 抓取成功。提取文本 {len(text_content)} 字，图片 {len(top_images)} 张。")
            # 将结果旁路写入黑板
            if state:
                state.add_source(title=title, url=url)
            # 组装返回给 Agent 的结果
            result = f"【网页抓取结果】\n网页标题: {title}\n\n"
            if top_images:
                result += f"【发现关键图片链接】(请务必调用 analyze_image 工具分析以下图片):\n"
                for i, img_url in enumerate(top_images):
                    result += f"图片 {i+1}: {img_url}\n"
                result += "\n"
            
            result += f"【网页正文内容摘要】:\n{text_content}"
            
            return result
        except Exception as e:
            logger.error(f"[Web Scraper] 抓取失败: {str(e)}")
            return f"抓取网页失败，错误原因: {str(e)}。这可能是由于目标网站有强反爬机制或网络超时。"