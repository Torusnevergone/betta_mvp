# app/core/llm_client.py
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
    tool_calls: Optional[dict] = None
    raw: Optional[dict] = None
class LLMClient:
    """
    Agent-ready LLM Client
    支持：
    - 普通 chat
    - 流式 stream
    - tool_calls 解析
    - 【新增】多模型路由（支持纯文本模型和视觉模型切换）
    """
    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.llm_api_key:
            raise ValueError("LLM_API_KEY 未配置：请在 .env 中设置")
        # 默认的文本模型客户端 (DeepSeek)
        self.text_endpoint = self.settings.llm_base_url.rstrip("/") + "/chat/completions"
        self.text_client = httpx.Client(
            timeout=self.settings.llm_timeout_s,
            headers={
                "Authorization": f"Bearer {self.settings.llm_api_key}",
                "Content-Type": "application/json",
            },
        )
        
        # 【新增】视觉模型客户端 (Qwen-VL)
        self.vision_endpoint = self.settings.vision_base_url.rstrip("/") + "/chat/completions"
        if self.settings.vision_api_key:
            self.vision_client = httpx.Client(
                timeout=self.settings.llm_timeout_s,
                headers={
                    "Authorization": f"Bearer {self.settings.vision_api_key}",
                    "Content-Type": "application/json",
                },
            )
        else:
            self.vision_client = None
    def _to_payload(self, messages: List[Message], stream: bool = False, tools: Optional[list] = None, use_vision: bool = False) -> dict:
        # 【修改点】增加 use_vision 参数，以便选择正确的 model 名称
        formatted_messages = []
        for m in messages:
            msg_dict = {"role": m.role}
            
            # 处理 assistant 带 tool_calls 的情况
            if m.role == "assistant" and m.tool_calls:
                msg_dict["content"] = m.content if m.content else None
            else:
                # 【修改点】这里做了一个小优化：如果 content 是列表（比如视觉模型需要的复杂结构），就不要强转为字符串
                if isinstance(m.content, list):
                    msg_dict["content"] = m.content
                else:
                    msg_dict["content"] = str(m.content) if m.content is not None else ""
            # 处理 tool_calls 结构（保持你原有的优秀逻辑不变）
            if m.tool_calls:
                safe_tool_calls = []
                for tc in m.tool_calls:
                    func_info = tc.get("function", {})
                    args = func_info.get("arguments", "{}")
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = str(args)
                    safe_tc = {
                        "id": str(tc.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": str(func_info.get("name","")),
                            "arguments": args
                        }
                    }
                    safe_tool_calls.append(safe_tc)
                msg_dict["tool_calls"] = safe_tool_calls
            # 处理 tool 角色
            if m.role == "tool":
                msg_dict["tool_call_id"] = str(m.tool_call_id) if m.tool_call_id else ""
                msg_dict["name"] = str(m.name) if m.name else "unknown_tool"
            formatted_messages.append(msg_dict)
        # 【修改点】根据 use_vision 决定使用哪个模型名称
        target_model = self.settings.vision_model if use_vision else self.settings.llm_model
        payload = {
            "model": target_model, 
            "messages": formatted_messages,
            "temperature": 0.3,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        
        return payload
    def chat(self, messages: List[Message], tools: Optional[list] = None, use_vision: bool = False) -> LLMResponse:
        """
        非流式调用。增加 use_vision 参数来切换模型。
        """
        payload = self._to_payload(messages, stream=False, tools=tools, use_vision=use_vision)
        
        # 根据 use_vision 选择对应的 client 和 endpoint
        client = self.vision_client if use_vision else self.text_client
        endpoint = self.vision_endpoint if use_vision else self.text_endpoint
        
        if not client:
            raise RuntimeError("未配置视觉模型的 API KEY，无法使用视觉功能！")
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
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
    def chat_stream(self, messages: List[Message], tools: Optional[list] = None, use_vision: bool = False) -> Iterable[str]:
        """
        流式调用。保持你原有的逻辑，只增加 use_vision 切换。
        """
        payload = self._to_payload(messages, stream=True, tools=tools, use_vision=use_vision)
        
        client = self.vision_client if use_vision else self.text_client
        endpoint = self.vision_endpoint if use_vision else self.text_endpoint
        with client.stream("POST", endpoint, json=payload) as r:
            r.raise_for_status()
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