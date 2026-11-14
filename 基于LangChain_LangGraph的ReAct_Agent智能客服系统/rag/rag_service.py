"""Desc :
总结服务类:用户提问,搜索参考资料,将提问和参考资料提交给模型,让模型根据提示词和参考资料总结回复
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from model.factory import chat_model
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts

def print_prompt(prompt):
    print(prompt.to_string())
    print("-"*20)
    return prompt

class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init__chain()

    def __init__chain(self):
        chain = self.prompt_template | RunnableLambda(print_prompt) |self.model | StrOutputParser()
        return chain

    # 接收用户查询query查询语句，调用检索器，召回相关文档
    def retriever_docs(self, query: str) ->list[Document] :
        return self.retriever.invoke(query)

    # rag总结函数
    def rag_summarize(self, query: str) -> str :
        # 把提取出来的参考资料拼接成字符串
        context_docs = self.retriever_docs(query)
        context = ""
        history = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}:参考资料:{doc.page_content} | 参考元数据:{doc.metadata}\n]"
        return self.chain.invoke(
            {"input": query,
             "context": context,
             "history": history
             })

if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合什么样的扫地机器人"))
