from __future__ import annotations
import json
from typing import List, Optional
from app.core.types import Message
from app.core.llm_client import LLMClient
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState

logger = setup_logger()

class BaseAgent:
    """
    符合 OpenAI Tool Calling 规范的 Agent 基类
    Agent 的抽象基类。
    它封装了“带工具的 LLM 调用”的核心逻辑。
    """

    def __init__(self, name: str, role_prompt: str, llm: LLMClient, tools: Optional[List[BaseTool]] = None):
        self.name = name
        self.role_prompt = role_prompt
        self.llm = llm
        self.tools = tools or []
        self._openai_tools = [tool.to_openai_schema() for tool in self.tools] if self.tools else None
        self._tool_map = {tool.name: tool for tool in self.tools}

    # 架构升级： chat方法接收state
    def chat(self, user_input:str, history:Optional[List[Message]]=None, state: Optional[SessionState] = None) -> str:
        messages = [Message(role="system", content=self.role_prompt)]
        if history:
            messages.extend(history)
        messages.append(Message(role="user",content=user_input))
        # logger.info(f"[{self.name}]正在思考...")
        # 真正的agent循环（解决llm认为工具调用不充足，多次调用工具的问题）
        max_iterations = 5 # 最多允许它连续调用3次工具

        for i in range(max_iterations):
            logger.info(f"[{self.name}]第{i+1}轮思考中...")
            # 调用大模型
            response = self.llm.chat(messages=messages, tools=self._openai_tools)
            # 安全修复：考虑情况tool_calls是None，不能for遍历
            tool_calls = response.tool_calls or []
            # 情况A：模型绝对不需要调用工具了（或者它已经总结完了)
            if not response.tool_calls:
                # 检查是否存在DSML标签
                if "<|DSML|function_calls>" in (response.content or ""):
                    logger.warning(f"[{self.name}] 发现模型输出了原生调用标签，强制截断并要求总结。")
                    messages.append(Message(role="assistant", content=response.content))
                    messages.append(Message(role="user", content="请停止调用工具，直接基于已有信息给出最终的中文总结。"))
                    continue # 再给它一次机会
                
                logger.info(f"[{self.name}] 思考完毕，给出最终回答。")
                return response.content or ""

            # 情况B：模型决定调用工具
            # 检查是否触发了工具调用
            # 行业规范：必须把 assistant 决定调用工具的这条消息，原封不动地加进历史记录中
            # 注意：此时 content 通常为 null 或空字符串
            messages.append(Message(
                role="assistant",
                content=response.content or "", # 确保content不是None
                tool_calls=tool_calls
            ))
            # 处理所有的工具调用 (模型可能一次性要求调用多个工具，比如同时搜两个词)
            for tool_call in response.tool_calls:
                call_id = tool_call.get("id", "call_default_id")
                # 安全获取function内部的name和arguments
                function_info = tool_call.get("function",{})
                # tool_name = tool_call["function"]["arguments"]
                tool_name = function_info.get("name", "unknown_tool")
                tool_args_str = function_info.get("arguments","{}")

                logger.info(f"[{self.name}]正在执行工具：{tool_name}，参数：{tool_args_str}")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}
                tool_result = "未找到该工具"
                if tool_name in self._tool_map:
                    tool_instance = self._tool_map[tool_name]
                    # 全局状态存储核心：执行工具时，把state传进去 
                    tool_result = tool_instance.run(state=state, **tool_args)
                    # tool_result = tool_instance.run(**tool_args)

                if not isinstance(tool_result, str):
                    tool_result = json.dumps(tool_result, ensure_ascii=False)

                # 行业规范：将工具执行的结果，以 role="tool" 的身份加进历史记录
                messages.append(Message(
                    role="tool",
                    content=tool_result,
                    tool_call_id=call_id,
                    name=tool_name
                ))
            logger.info(f"[{self.name}] 工具执行完毕，正在整合结果...")

        # 如果循环了3次还没结束，强制总结
        logger.warning(f"[{self.name}] 达到最大思考轮数 ({max_iterations})，强制停止。")
        # messages.append(Message(role="user", content="搜索轮数已达上限，请立即基于上面的所有搜索结果给出最终总结。不要再调用任何工具。"))
        # 防止insightagent不调用情感分析工具，在最大轮次时产生道歉
        messages.append(Message(
            role="user",
            content="【系统警告】：搜索轮数已达上限，必须立即停止调用任何工具！请直接基于现有信息给出最终结论。注意：你必须且只能输出符合你角色要求的纯 JSON 格式，绝不允许输出任何解释性文字或道歉！"
        ))
        final_response = self.llm.chat(messages=messages, tools=None)
        return final_response.content or ""


