from __future__ import annotations
import json
import sqlite3
from typing import Optional
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState
from app.core.llm_client import LLMClient
from app.core.types import Message

logger = setup_logger()

class NL2SQLTool(BaseTool):
    name = "query_business_db"
    description = "当需要查询公司内部的客诉记录、销售数据或退换货订单时，调用此工具。输入自然语言问题，系统会自动将其转化为 SQL 语句并查询数据库。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要查询的具体业务问题，例如 '统计4月份关于车机卡死的客诉数量'"
            }
        },
        "required": ["query"]
    }

    def __init__(self, llm:LLMClient, db_path:str="app/storage/business.db"):
        self.llm = llm
        self.db_path = db_path
        # 核心：告诉他模型我们的数据库结构(Schema)
        self.db_schema = """
        表名: customer_complaints (客诉记录表)
        字段:
        - id (INTEGER): 主键
        - product_model (TEXT): 产品型号 (如 'SU7 Max', 'SU7 Pro')
        - issue_type (TEXT): 问题类型 (如 '车机卡死', '漆面划痕')
        - status (TEXT): 处理状态 ('处理中', '已解决', '待处理')
        - complaint_date (DATE): 投诉日期 (格式 'YYYY-MM-DD')
        """
    def run(self, state:Optional[SessionState] = None, **kwargs) -> str:
        user_query = kwargs.get("query","")
        if not user_query:
            return json.dumps({"error": "查询词不能为空"})
        logger.info(f"[Tool Call] 正在执行 NL2SQL, 业务需求: 【{user_query}】")

        # 1.用LLM将自然语言翻译为SQL
        prompt = (
            f"你是一个精通 SQLite 的数据库专家。\n"
            f"这是我的数据库结构：\n{self.db_schema}\n\n"
            f"用户的需求是：{user_query}\n"
            f"请写出对应的 SQL 查询语句。只输出纯 SQL 语句，绝对不要输出任何其他解释性文字，不要使用 Markdown 代码块。"
        )
        sql_query = self.llm.chat([Message(role="user", content=prompt)]).content.strip()
        # 清理可能带上的markdown标签
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        logger.info(f"[NL2SQL] 生成的 SQL 语句: {sql_query}")

        # 2.执行SQL并获取结果
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            # 获取列名
            col_names = [description[0] for description in cursor.description]
            conn.close()

            # 3.组装结果给agent
            if not results:
                return f"执行 SQL: {sql_query}\n结果: 未查询到相关数据。"
            
            # 格式化成字典列表
            formatted_results = [dict(zip(col_names, row)) for row in results]
            output = f"执行 SQL: {sql_query}\n查询结果:\n{json.dumps(formatted_results, ensure_ascii=False, indent=2)}"

            # 旁路记录url
            if state:
                state.add_source(title="公司内部客诉数据库查询结果", url="local://business_db")

            return output

        except Exception as e:
            logger.error(f"[NL2SQL] 数据库查询失败: {e}")
            return f"执行 SQL 失败: {sql_query}\n错误信息: {str(e)}"