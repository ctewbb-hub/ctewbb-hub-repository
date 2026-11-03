"""Desc :

"""
import time
import streamlit as st
from rag import RagService
from config_data import session_config

st.title("智能客服")
st.divider()    # 题目下方的分隔符

if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好,我是通义千问,有什么可以帮助你的?"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()
# 显示同一个状态下的历史记录
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
# 页面下方提供用户输入栏位
prompt = st.chat_input()
if prompt:
    # 聊天中显示用户的提问内容
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    ai_res_list= []
    with st.spinner("AI思考中..."):
        res_stream = st.session_state["rag"].chain.stream({"input":prompt}, session_config)

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                # yield 把普通函数变成「生成器」，函数执行到 yield 时，暂停，返回这个值；下次调用继续从暂停的地方接着跑，不是从头重来。
                yield chunk
        st.chat_message("assistant").write(capture(res_stream,ai_res_list))
        # res_stream是一个迭代器,此时同意状态的历史记录没有保存下来
        # "".join()是字符串方法，括号里放一个可迭代对象（列表、生成器等，里面每一项都必须是字符串）,用分隔符把元素连起来,变成一个字符串
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})





