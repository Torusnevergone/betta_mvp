import sqlite3
import os
from typing import List
from app.core.types import Message
from app.core.logging import setup_logger

logger = setup_logger()

# 这里我们使用 SQLite 模拟 MySQL。
# 生产环境中，只需把这里的 sqlite3 换成 pymysql 或 SQLAlchemy 即可。
DB_PATH = "app/storage/chat_history.db"
def init_db():
    """初始化数据库表结构"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("[Database] 聊天记录表初始化成功。")

def save_message(session_id:str, role:str, content:str):
    """保存单条消息到数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ? ,?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def get_recent_history(session_id:str, limit:int = 6) -> List[Message]:
    """
    从数据库获取最近的 N 条聊天记录（滑动窗口的核心）。
    返回的列表已转换为 Message 对象，且按时间正序排列。
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # 按照时间倒序，获取最近的limit条记录
    cursor.execute('''
        SELECT role, content FROM messages 
        WHERE session_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    ''',(session_id, limit))

    results = cursor.fetchall()
    conn.close()

    if not results:
        return []

    
    # result是倒序的，把它反转回来
    history = []
    for role, content in reversed(results):
        history.append(Message(role=role, content=content))

    return history