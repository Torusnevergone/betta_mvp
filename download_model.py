# download_model.py
import os
# 依然使用镜像源下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from sentence_transformers import SentenceTransformer
print("开始下载模型到本地 models/ 目录...")
# 加载模型
model = SentenceTransformer('shibing624/text2vec-base-chinese')
# 将模型保存到本地目录
save_path = os.path.join(os.getcwd(), "models", "text2vec-base-chinese")
os.makedirs(save_path, exist_ok=True)
model.save(save_path)
print(f"模型已成功下载并保存在: {save_path}")
print("以后代码里直接读取这个本地路径，再也不会有网络报错了！")