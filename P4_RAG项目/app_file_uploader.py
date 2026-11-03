"""Desc :
基于streamlit开发的web网页开发
streamlit架构的问题:每次网页刷新/状态改变,代码就会重新运行,无法保证状态的连贯,
例如本案例想统计上传文件数量的变化,由于代码每次都会重新运行,数量始终无法叠加,为此架构提供解决方案:st.session_state
"""
import streamlit as st
from knowledge_base import KnowledgeBaseService
import time
# 添加网页标题
st.title("知识库更新服务")

# 增加文件上传服务
uploader_file = st.file_uploader(
    "请上传txt文件",
    type = ["txt"],
    accept_multiple_files= False,   # 是否支持多文件一块上传
)
# st.session_station是官方提供的字典,里面的信息不会随着网页状态刷新而刷新
if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # 提取文件信息
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size/1024    #KB

    st.subheader(f"文件名：{file_name}")
    st.write(f"文件格式：{file_type},大小：{file_size}KB")

    # 得到提取文件的内容getvalue->bytes->decode('utf-8')
    text = uploader_file.getvalue().decode("utf-8")
    # st.write(f"文件内容如下：\n{text}")

    with (st.spinner("载入知识库中...")):  # 在spinner内的代码执行过程中会有一个转圈动画
        time.sleep(1)
        result = st.session_state["service"].upload_by_str(data=text, filename=file_name)
        st.write(result)
        # print(file_name)
