from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SessionState:
    """
    全局会话状态（黑板）。
    用于在各个 Agent 之间传递非语义数据（如 URL、元数据），
    避免将这些数据塞入 LLM 的 Prompt 中造成浪费和幻觉。
    """
    topic:str = ""

    # 旁路存储：存储所有的来源链接(LLM看不到)
    sources: List[Dict[str,str]] = field(default_factory=list)

    # 论坛机制： 记录论坛所有人发言
    forum_logs:List[str] = field(default_factory=list)

    def add_source(self, title:str, url:str):
        """添加旁路来源并去重"""
        # self.evidence_texts.append(text)

        # 去重逻辑：如果这个URL还没用被存过，就存进去
        if not any(s.get("url") == url for s in self.sources):
            self.sources.append({
                "title": title,
                "url": url
            })

    # 论坛机制：新增函数
    def add_chat_record(self, agent_name:str, content:str):
        self.forum_logs.append(f"【{agent_name}】: {content}")

    def get_forum_context(self) -> str:
        if not self.forum_logs:
            return "论坛目前无发言"
        return "\n\n".join(self.forum_logs)
    
    