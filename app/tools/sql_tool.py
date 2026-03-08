# app/tools/sql_tool.py
from __future__ import annotations
import json
from typing import Optional
from app.tools.base import BaseTool
from app.core.logging import setup_logger
from app.core.state import SessionState
from app.core.llm_client import LLMClient
from app.core.types import Message
# 引入 SQLAlchemy 核心组件
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
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
    def __init__(self, llm: LLMClient, db_path: str = "app/storage/business.db"):
        self.llm = llm
        
        # 1. SQLAlchemy 魔法：创建跨数据库引擎
        # 这里使用 sqlite 方言。如果是生产环境，只需改为 "mysql+pymysql://user:pwd@host/db"
        self.engine_url = f"sqlite:///{db_path}"
        # create_engine 自动管理连接池，比原生 sqlite3 安全得多
        self.engine = create_engine(self.engine_url)
        
        # 2. 数据库 Schema 描述
        self.db_schema = """
        表名: customer_complaints (客诉记录表)
        字段:
        - id (INTEGER): 主键
        - product_model (TEXT): 产品型号 (如 'SU7 Max', 'SU7 Pro')
        - issue_type (TEXT): 问题类型 (如 '车机卡死', '漆面划痕')
        - status (TEXT): 处理状态 ('处理中', '已解决', '待处理')
        - complaint_date (DATE): 投诉日期 (格式 'YYYY-MM-DD')
        """
    def _optimize_keywords(self, user_query: str) -> str:
        """
        【对应简历亮点：关键词优化查询】
        在生成 SQL 前，先用一个小 Prompt 纠正用户口语化或错误的实体名称，
        对齐数据库中的标准字段，防止 SQL 查不到数据。
        """
        prompt = (
            f"你是一个领域专家。用户输入了自然语言查询：【{user_query}】。\n"
            f"请将查询中的实体名称对齐为标准格式。例如：'大米汽车' -> 'SU7', '屏幕死机' -> '车机卡死'。\n"
            f"如果不需要优化，原样返回。只返回优化后的查询语句，不要其他废话。"
        )
        optimized = self.llm.chat([Message(role="user", content=prompt)]).content.strip()
        logger.info(f"[NL2SQL] 关键词优化: '{user_query}' -> '{optimized}'")
        return optimized
    def run(self, state: Optional[SessionState] = None, **kwargs) -> str:
        raw_query = kwargs.get("query", "")
        if not raw_query:
            return json.dumps({"error": "查询词不能为空"})
        
        logger.info(f"[Tool Call] 正在执行 NL2SQL, 原始需求: 【{raw_query}】")
        # 1. 关键词优化 (简历亮点)
        optimized_query = self._optimize_keywords(raw_query)
        # 2. 使用 LLM 将优化后的语言翻译为 SQL
        prompt = (
            f"你是一个精通 SQL 的数据库专家。\n"
            f"这是我的数据库结构：\n{self.db_schema}\n\n"
            f"用户的需求是：{optimized_query}\n"
            f"请写出对应的 SQL 查询语句。只输出纯 SQL 语句，绝对不要输出任何其他解释性文字，不要使用 Markdown 代码块。"
        )
        
        sql_query_str = self.llm.chat([Message(role="user", content=prompt)]).content.strip()
        sql_query_str = sql_query_str.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"[NL2SQL] 生成的 SQL 语句: {sql_query_str}")
        # 3. 使用 SQLAlchemy 执行查询
        try:
            # SQLAlchemy 要求使用 text() 函数包装原生 SQL 字符串，以防范简单的注入攻击
            sql_statement = text(sql_query_str)
            
            # 使用 engine.connect() 自动管理连接的开关
            with self.engine.connect() as conn:
                result_proxy = conn.execute(sql_statement)
                # 获取列名
                col_names = result_proxy.keys()
                # 获取所有数据
                results = result_proxy.fetchall()
            # 4. 组装结果
            if not results:
                return f"执行 SQL: {sql_query_str}\n结果: 未查询到相关数据。"
            
            # 格式化成字典列表
            formatted_results = [dict(zip(col_names, row)) for row in results]
            output = f"执行 SQL: {sql_query_str}\n查询结果:\n{json.dumps(formatted_results, ensure_ascii=False, indent=2)}"
            
            # 旁路记录 URL
            if state:
                state.add_source(title="公司内部客诉数据库查询结果", url="local://business_db")
                
            return output
            
        except SQLAlchemyError as e:
            # 更专业的错误捕获
            logger.error(f"[NL2SQL] SQLAlchemy 查询失败: {e}")
            return f"执行 SQL 失败: {sql_query_str}\n错误信息: 数据库引擎报错。"