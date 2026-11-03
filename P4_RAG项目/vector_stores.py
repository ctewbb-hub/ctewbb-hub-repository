"""Desc :

"""
from langchain_chroma import Chroma
import config_data as config

class VectorStoreService(object):
    def __init__(self, embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )
    # 返回向量检索器,方便加入chain
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})  # similarity_score_threshold的最小相关性阈值

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings()).get_retriever()
    # 可以看到as_retriever返回VectorStoreRetriever对象,该对象又继承自Runnable,所以可以用invoke方法,但是retriever实例对象进入invoke发现进行了@override重写,invoke方法中input传str即可
    res = retriever.invoke("我的体重180斤,尺码推荐")
    print(res)