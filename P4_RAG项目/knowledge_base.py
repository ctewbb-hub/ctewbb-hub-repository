"""Desc :
知识库
"""
import os
import hashlib # 计算字符串的md5值-md5 算法:不管字符串文本多长计算出来的md5代码永远是32位的16进制代码
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import config_data as config
from langchain_chroma import Chroma

def get_string_md5(input_str,encoding="utf-8"):
    """将传入的字符串转化为md5字符串"""

    # 将字符串转换为bytes字节数组
    str_bytes = input_str.encode(encoding=encoding)

    # 创建md5对象并将bytes数组转化为md5
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()# 得到md5的16进制字符串

def save_md5(md5_str):
    """将传入的md5字符串记录到文件保存"""
    with open(config.md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str + "\n")

def check_md5(md5_str):
    """检查传入的md5字符串是否已经被处理过，传入的数据库文件生成md5符号传入md5.text文件，如果数据库中已经有相同的文件，则新传入的文件比照md5之后不用处理"""
    if not os.path.exists(config.md5_path):
        # 如果md5文件不存在,表示还没有处理过文件
        open(config.md5_path, 'w' ,encoding= "utf-8").close()   # w模式打开文件没有自动创建
        return False
    else:
        with open(config.md5_path, 'r' ,encoding= "utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if line== md5_str:
                    return True
        return False

class KnowledgeBaseService(object):
    def __init__(self):
        os.makedirs(config.persist_directory,exist_ok=True) # 如果文件夹不存在创建,存在跳过
        self.chroma =  Chroma(
            collection_name=config.collection_name, # 数据库的表名
            embedding_function=DashScopeEmbeddings(),
            persist_directory=config.persist_directory, #数据库的本地存储文件夹
        ) # 向量存储的实例:Chroma向量数据库对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,   #分割后的每一个文本段最大长度
            chunk_overlap= config.chunk_overlap,    # 连续文本段之间的字符重叠数量
            separators=config.separators,   # 自然段落划分的符号
            # length_limit=len,   # 使用python自带的len函数做长度统计的依据
        ) #递归文本分割器对象

    def upload_by_str(self,data,filename):
        """将传入的字符串进行向量化,传入向量数据库"""
        md5_hex = get_string_md5(data)

        if check_md5(md5_hex):
            return "[跳过],内容已经存在数据库中"
        else:
            knowledge_chunks: list[str] = self.spliter.split_text(data)

        metadata = {
            "source" : filename,
            "create_time"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator" : "Mr.chen"
        }

        self.chroma.add_texts(
            knowledge_chunks,
            metadatas=[metadata for _ in knowledge_chunks], # 按照切片的knowledge_chunks的大小创建同样大小的list存储数据
        )

        save_md5(md5_hex)
        return "[成功],内容已经成功载入向量库"

if __name__ == "__main__":
    service = KnowledgeBaseService()
    r = service.upload_by_str("陈先生","textfile")
    print(r)