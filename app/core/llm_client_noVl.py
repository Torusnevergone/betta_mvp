from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import httpx
import json

from app.core.config import get_settings
from app.core.types import Message

@dataclass
class LLMResponse:
    content: str
    # text: str 
    # 不再只支持text
    tool_calls: Optional[dict] = None
    raw: Optional[dict] = None


class LLMClient:
    """
    DeepSeek（OpenAI风格）Chat Completions 客户端。

    为什么要封装在一个类里：
    - 上层只关心 chat(messages) -> text
    - 以后换 OpenAI/通义/智谱，只需要改这里
    """
    
    """
    Agent-ready LLM Client
    支持：
    - 普通 chat
    - 流式 stream
    - tool_calls 解析
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.llm_api_key:
            raise ValueError("LLM_API_KEY 未配置：请在 .env 中设置")

        self.endpoint = self.settings.llm_base_url.rstrip("/") + "/chat/completions"

        self.client = httpx.Client(
            timeout = self.settings.llm_timeout_s,
            headers = {
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )
    
    def _to_payload(self, messages: List[Message], stream: bool = False, tools: Optional[list] = None) -> dict:
        # 因为 Message 增加了字段，我们在把 Message 转成发给 API 的 payload 时，需要把这些字段加上
        formatted_messages = []
        for m in messages:
            # msg_dict = {"role": m.role, "content": m.content}
            # 基础结构
            msg_dict = {"role": m.role}
            # OpenAI规范：如果role是assistant且有tool_calls, content最好是None而不是空字符串
            # DeepSeek（和 OpenAI）对 role="assistant" 带 tool_calls 的消息有极其严格的格式要求：tool_calls 必须是完整的字典结构，不能有遗漏。
            if m.role == "assistant" and m.tool_calls:
                msg_dict["content"] = m.content if m.content else None
            else:
                # 确保content是字符串
                # msg_dict["content"] = m.content
                msg_dict["content"] = str(m.content) if m.content is not None else ""
            


            '''
            # 动态组装字段，避免传递引发 400 的 null 或空字符串
            # 只有当 content 真的有实质内容时，我们才传它
            # 或者当它是一条普通消息（没有 tool_calls）时，我们才传它
            if m.content and m.content.strip():
                msg_dict["content"] = str(m.content)
            elif not m.tool_calls and m.role != "tool":
                # 普通对话，即使是空的，也必须传 content
                msg_dict["content"] = ""
            '''


            # 如果是assistant发出的工具调用请求
            if m.tool_calls:
                # 必须严格重构tool_calls结构，丢弃多余内部字段
                safe_tool_calls = []
                for tc in m.tool_calls:
                    # 规范修复 arguments一定是JSON字符串而非字典
                    func_info = tc.get("function", {})
                    args = func_info.get("arguments", "{}")
                    # 致命修复：如果 arguments 变成了字典，强制转回 JSON 字符串
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = str(args)

                    safe_tc = {
                        # "id": tc.get("id",""),
                        "id": str(tc.get("id", "")),
                        "type": "function",
                        "function": {
                            #"name": tc.get("function", {}).get("name", ""),
                            "name": str(func_info.get("name","")),
                            # "arguments": tc.get("function", {}).get("arguments","{}")
                            "arguments": args
                        }
                    }
                    safe_tool_calls.append(safe_tc)
                msg_dict["tool_calls"] = safe_tool_calls

            # 如果是返回给模型的工具执行结果
            if m.role == "tool":
                # msg_dict["tool_call_id"] = m.tool_call_id
                msg_dict["tool_call_id"] = str(m.tool_call_id) if m.tool_call_id else ""
                # 注意：OpenAI 规范中，role="tool" 时，name 字段也是必须的
                # msg_dict["name"] = m.name
                msg_dict["name"] = str(m.name) if m.name else "unknown_tool"
                # 工具的content必须传，哪怕是空的
                # msg_dict["content"] = str(m.content) if m.content else "空结果"

            formatted_messages.append(msg_dict)
            
            '''
            # 如果是 assistant 发出的工具调用请求
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            # 如果是我们返回给模型的工具执行结果
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            formatted_messages.append(msg_dict)
            '''

        paylaod = {
            "model": self.settings.llm_model, 
            # "messages": [{"role": m.role, "content": m.content} for m in messages],
            "messages": formatted_messages,
            "temperature": 0.3,
            "stream": stream,
        }

        if tools:
            paylaod["tools"] = tools
        
        return paylaod
        

    def chat(self, messages: List[Message], tools: Optional[list] = None) -> LLMResponse:
        """
        非流式，一次性拿完整文本
        """
        payload = self._to_payload(messages, stream=False, tools=tools)

        # 一个简单的“轻量重试”： 网络抖动/限流时更稳
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = self.client.post(self.endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()

                # OpenAI风格： choices[0].message.content
                # text = data["choices"][0]["message"]["content"]
                choice = data["choices"][0]
                message = choice["message"]

                content = message.get("content", "")
                tools_calls = message.get("tool_calls")
                
                return LLMResponse(
                    content=content,
                    tool_calls=tools_calls,
                    raw=data,
                )
            except Exception as e:
                last_err = e
        
        raise RuntimeError(f"LLM 调用失败（重试3次仍失败）：{last_err}")

    def chat_stream(self, messages: List[Message], tools: Optional[list] = None) -> Iterable[str]:
        """
        流式：边生成边吐（后面做 SSE/前端显示会用到）
        这里先给结构；具体 DeepSeek 的流式 chunk 格式我们下一步再精化。
        """
        payload = self._to_payload(messages, stream=True, tools=tools)

        with self.client.stream("POST", self.endpoint, json=payload) as r:
            r.raise_for_status()
            # 兼容 SSE 风格：逐行读取
            for line in r.iter_lines():
                if not line:
                    continue
                
                if line.startswith("data: "):
                    line = line[len("data: "):]

                if line.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(line)
                    delta = data["choices"][0]["delta"]

                    if "content" in delta:
                        yield delta["content"]
                
                except Exception:
                    continue
                # 不同兼容接口的 stream 格式会有差异：
                # 我们先把原始行吐出去，下一步再解析成 token 文本
                # yield line
        # “为什么 stream 先不解析成 token”：不同厂商的流式返回字段略不同，先把管道搭通，后面我们再做“只输出增量 content”


