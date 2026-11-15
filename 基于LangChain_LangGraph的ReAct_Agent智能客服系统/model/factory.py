"""Desc :
提供模型
"""
from abc import abstractmethod, ABC
from typing import Optional
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_config
# 返回模型的父类
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel

class BaseModelFactory(ABC):
    @abstractmethod     # 描述:抽象方法
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return ChatTongyi(model=rag_config["chat_model_name"])


class EmbeddingsModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel | Embeddings]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"])

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsModelFactory().generator()