"""Desc :

"""
import streamlit as st
from agent.react_agent import ReactAgent

st.set_page_config(page_title="机器人智能客服", page_icon="🤖")
st.title("🤖 机器人智能客服")
st.caption("基于 LangChain ReActAgent + RAG 检索增强")
st.divider()    # 分隔符

# 保存agent实例对象
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
# 保存历史聊天记录
if "messages" not in st.session_state:
    st.session_state["messages"] = []
# 页面刷新 / 重新运行的时候，把存在会话状态里的历史聊天记录，重新渲染展示在聊天界面上，实现「多轮对话历史不消失」。
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    response_messages: list[str] = []
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def stream_generator(generator, cache_list):
            """逐字流式输出，同时缓存完整响应(AI回复显示)"""
            for chunk in generator:
                cache_list.append(chunk)
                # 一行一行返回更细致化,一个字符一个字符得返回
                for char in chunk:
                    yield char  # yield把普通函数变成【生成器】，函数可以中途暂停、返回一个值，下次调用从暂停的地方继续跑
        # 用户提问和AI回复都会显示
        st.chat_message("assistant").write_stream(stream_generator(res_stream, response_messages))
        st.session_state["messages"].append({"role": "assistant", "content": "".join(response_messages)})
        st.rerun()  # 每次状态变化自动让页面重新刷新