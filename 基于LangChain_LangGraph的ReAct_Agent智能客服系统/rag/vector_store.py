"""Desc :
文档的分割,md5以及导入chroma向量库
"""
import os
from os.path import split
# from typing import List
from langchain_chroma import Chroma
# 导入文档与文档分隔 RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 导入文件处理模块
from utils.file_handler import pdf_loader, txt_loader, get_file_md5_hex, listdir_with_allowed_type
# 导入配置文件
from utils.config_handler import chroma_config
# 向量库服务类
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
# 导入模型
from model.factory import embed_model

class VectorStoreService:
    def __init__(self):

        self.vector_store = Chroma(
            collection_name=chroma_config["collection_name"],  # 相当于设置数据库名
            embedding_function=embed_model,  # 使用的嵌入模型
            persist_directory=get_abs_path(chroma_config["persist_directory"]),  # 数据库位置
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_config["chunk_size"],
            chunk_overlap=chroma_config["chunk_overlap"],
            separators=chroma_config["separators"]
        )
    # 获取检索器
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_config["k"]})

    def load_document(self):
        """
        从数据文件夹中读取数据文件，转为向量存入向量
        同时还需要做MD5去重，已经在向量库中则不需要加载
        """
        def check_md5_hex(md5):
            # 判断md5文件是否存在
            if not os.path.exists(get_abs_path(chroma_config["md5_hex_store"])):
                # 不存在则创建
                open(get_abs_path(chroma_config["md5_hex_store"]), "w" ,encoding="utf-8").close()

            # md5存在则读文档对比传入的md5
            with open(get_abs_path(chroma_config["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    if line.strip() == md5:
                        return True     # md5处理过
                return False    # md5没处理过

        def save_md5_hex(md5_for_check : str):
            with open(get_abs_path(chroma_config["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        # 判断文件类型
        def get_file_documents(read_path : str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith(".txt"):
                return txt_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_config["data_path"]),
            tuple(chroma_config["allow_knowledge_file_type"])
        )
        for path in allowed_files_path:
            # 获取文件的md5
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在于知识库内,跳过")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容(文件内容为空)")
                    continue
                split_document : list[Document]= self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容(分片后文件内容为空)")
                    continue
                # 将内容存入向量库
                self.vector_store.add_documents(split_document)
                # 记录这个已经处理好的文件的md5,避免下次重复加载
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as e:
                # exc_info=True:详细记录报错的堆栈,false:仅记录报错内容
                logger.error(f"[加载知识库]{path}加载失败:{str(e)}", exc_info=True)
                continue

if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("充电")
    for r in res:
        print(r.page_content)
        print("*" * 20)