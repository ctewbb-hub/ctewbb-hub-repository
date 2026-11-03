"""Desc :

"""
# embedding_model和chat_model配置
embedding_model = "text-embedding-v1"
chat_model = "qwen-plus"

# md5文件路径配置
md5_path = "./md5.text"

# Chroma配置
collection_name = "rag"
persist_directory = "./chroma.db"

# spliter配置
chunk_size = 1000
chunk_overlap = 100
separators = ["\n",",","!","?","."]

# 向量检索器配置
similarity_threshold = 1 # 检索返回匹配的文档数量

# session_config配置
session_config = {"configurable":{"session_id":"user001"}}
