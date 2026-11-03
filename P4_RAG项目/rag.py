"""Desc :

"""
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
import config_data as config
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(embedding=DashScopeEmbeddings(model=config.embedding_model))
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的参考资料为主,简介专业得回答用户问题,参考资料:{context}"),
            ("system","并且我提供的用户对话历史记录如下"),
            MessagesPlaceholder("history"),
            ("user", "回答用户提问:{input}")
        ])
        self.chat_model = ChatTongyi(model=config.chat_model)
        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        chain = (
                RunnablePassthrough.assign(context=itemgetter("input") | retriever)
                | self.prompt_template
                | self.chat_model
                | StrOutputParser()
        )
        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

if __name__ == '__main__':
    # session_id 固定配置格式
    session_config = {"configurable":{"session_id":"user001"}}
    res = RagService().chain.invoke({"input":"我的体重58kg,尺码推荐"}, session_config)
    print(res)