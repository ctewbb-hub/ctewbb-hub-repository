"""Desc :

"""
import os,json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import messages_from_dict, BaseMessage, message_to_dict

def get_history(session_id):
    return FileChatMessageHistory("./chat_history",session_id)

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, storage_path, session_id):
        super().__init__()
        self.storage_path = storage_path    # 不同会话id文件的存储路径
        self.session_id = session_id    # 会话id
        self.file_path = os.path.join(self.storage_path, f"{self.session_id}.json")
        # 确保文件夹存在(os.path.dirname会把文件夹而不是文件建好,文件留给open创建)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    @property  # 装饰器,将messages方法变为成员属性来用
    def messages(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)  # 从打开的文件读取json内容,此处内容为list[字典]注意json.load()和json.loads()的区别
                return messages_from_dict(messages_data)# list[BaseMessages]
        except FileNotFoundError:
            return []

    def add_message(self, message: BaseMessage) -> None:
        # messages:list[message],可以self.message进去看
        # 复制一份给all_messages
        all_messages = list(self.messages)
        all_messages.append(message)
        new_messages = [message_to_dict(message) for message in all_messages]
        # 将数据改成json格式写入文件:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(new_messages,f,ensure_ascii=False, indent=2)

    # 可选批量接口，框架会优先调用add_message
    def add_messages(self, messages) -> None:
        for m in messages:
            self.add_message(m)

    def clear(self):
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump([], f ,ensure_ascii=False, indent=2)