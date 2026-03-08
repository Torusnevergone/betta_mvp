from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.logging import setup_logger
from app.core.state import SessionState

# 引入向量化模型和FAISS库
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.tools.base import BaseTool

logger = setup_logger()

class LocalRAGTool(BaseTool):
    """
    本地知识库检索工具 (RAG)。
    负责将用户的查询转化为向量，并在本地文档中寻找最匹配的段落。
    """
    name = "local_rag_search"
    description = "当需要查找公司内部文档、私有知识库、历史预案或非公开数据时，调用此工具。输入查询词，返回高度相关的文档片段。"

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要查询的具体问题，例如 '公司对SU7起火事故的公关预案是什么？'"
            },
            "top_k": {
            "type": "integer",
            "description": "期望返回的相关段落数量，默认为 3",
            "default": 3
            }
        },
        "required": ["query"]
    }

    def __init__(self, docs_dir: str = "docs"):
        """
        初始化 RAG 工具。
        在 MVP 阶段，我们每次初始化时动态读取 docs 目录下的 txt 文件并构建索引。
        【行业规范提示】：在生产环境中，构建索引是异步/离线的，这里仅为演示。
        """
        self.docs_dir = Path(docs_dir)
        self.chunks: List[Dict[str, str]] = [] # 保存文本段落和来源 向量与负载（Payload/Metadata）分离存储

        # 1. 加载一个轻量级的中文向量模型 (首次运行会自动下载)
        logger.info("[RAG] 正在加载 Embedding 模型 (首次可能较慢)...")
        self.encoder = SentenceTransformer('shibing624/text2vec-base-chinese')
        # 离线加载
        # self.encoder = SentenceTransformer('./models/text2vec-base-chinese')

        # 2. 读取并切分文档
        self._load_and_chunk_documents()

        # 3.构建向量库
        if self.chunks:
            logger.info(f"[RAG] 正在构建向量索引，共 {len(self.chunks)} 个文本块...")
            # 把所有文本变成向量
            texts = [chunk["text"] for chunk in self.chunks]
            embeddings = self.encoder.encode(texts)

            # FAISS初始化：获取向量维度大小，并建立索引
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension) # 使用L2距离（欧氏距离）进行相似度计算
            self.index.add(np.array(embeddings).astype('float32'))
            logger.info("[RAG] 本地知识库加载完毕！")
        else:
            self.index = None
            logger.warning("[RAG] docs 目录下没有找到 txt 文件，知识库为空。")

    def _load_and_chunk_documents(self):
        """
        极简的分块 (Chunking) 逻辑。
        按行切分 txt 文件。
        行业规范是使用“递归字符切分（Recursive Character Splitter）”：
        先尝试按段落切（\n\n），如果切出来的块还是太大（比如超过 500 字），
        就按句子切（。 或 .），再大就按逗号切，
        并且相邻的两个块之间要保留一定的重叠（Overlap，比如 50 个字），
        防止把一句话从中间硬生生切断导致语义丢失。
        """
        if not self.docs_dir.exists():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            return
        for file_path in self.docs_dir.glob("*.txt"):   # glob用于模式匹配查找文件
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # 极其简单的切分：按空行切分成段落
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 10]
                for p in paragraphs:
                    self.chunks.append({
                        "text": p,
                        "source": f"Local Doc: {file_path.name}"
                    })
    
    # 加入state
    def run(self, state: Optional[SessionState] = None, **kwargs) -> str:
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 3)
        if not query:
            return json.dumps({"error": "查询词不能为空"})
        
        if not self.index or len(self.chunks) == 0:
            return json.dumps({"error": "本地知识库为空，无法检索"})
        logger.info(f"[Tool Call] 正在执行 local_rag_search, 查询: 【{query}】")

        # 把用户的查询变成向量
        query_vector = self.encoder.encode([query])

        # 在FAISS库中搜索最详尽的top_k个向量
        # D是距离数组，I是对应的索引数组
        D, I = self.index.search(np.array(query_vector).astype('float32'), min(top_k, len(self.chunks)))
        # search: 这是 FAISS 库底层用 C++ 写的一个极其高效的相似度计算函数。

        # 根据索引把原本捞出来
        '''
        results_evidence = []
        for idx in I[0]:
            if idx != -1: # FAISS没找到是会返回-1
                chunks = self.chunks[idx]
                ev = {
                    "source": chunks["source"],
                    "title": "内部文档片段",
                    "url": "loacl://docs", # 伪造一个本地URL格式，方便后续提取。在正规的知识库系统中，本地文档的引用通常使用 URI (统一资源标识符)，而不是 URL（统一资源定位符）。
                    "snippet": chunks["text"]
                }
                results_evidence.append(ev)
        return json.dumps(results_evidence, ensure_ascii=False)
        '''

        # 加入state后
        llm_readable_results = []
        for i, idx in enumerate(I[0]):
            if idx != -1:
                chunk = self.chunks[idx]
                llm_readable_results.append(f"[内部文档 {i+1}] {chunk['text']}")

                # 旁路写入
                if state:
                    state.add_source(title=chunk["source"], url="local://docs")
        return "\n\n".join(llm_readable_results)