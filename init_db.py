# init_db.py
import sqlite3
import os
def init_business_db():
    # 确保 storage 目录存在
    os.makedirs("app/storage", exist_ok=True)
    db_path = "app/storage/business.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建客诉工单表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customer_complaints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_model TEXT,
        issue_type TEXT,
        status TEXT,
        complaint_date DATE
    )
    ''')
    
    # 清空旧数据（防止重复运行插入多次）
    cursor.execute('DELETE FROM customer_complaints')
    
    # 插入模拟数据 (小米SU7的客诉记录)
    complaints = [
        ('SU7 Max', '车机卡死', '处理中', '2024-04-01'),
        ('SU7 Pro', '漆面划痕', '已解决', '2024-04-02'),
        ('SU7 Max', '辅助驾驶失灵', '待处理', '2024-04-05'),
        ('SU7 标准版', '车机卡死', '已解决', '2024-04-06'),
        ('极氪007', '异响', '已解决', '2024-03-28')
    ]
    cursor.executemany('INSERT INTO customer_complaints (product_model, issue_type, status, complaint_date) VALUES (?, ?, ?, ?)', complaints)
    
    conn.commit()
    conn.close()
    print(f"业务数据库已生成: {db_path}，包含 {len(complaints)} 条记录。")
if __name__ == "__main__":
    init_business_db()