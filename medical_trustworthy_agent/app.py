# app.py
import streamlit as st
from modules.workflow import MedicalTrustworthyAgent
from utils.ui_streamlit import render_streamlit_ui
from utils.visualization import format_agent_result

# ------------------------------
# 初始化 Agent（启动时加载模型）
# ------------------------------
@st.cache_resource(show_spinner=True)
def init_agent():
    agent = MedicalTrustworthyAgent(config_path="config.yaml")
    return agent

agent = init_agent()

# ------------------------------
# 用户输入
# ------------------------------
st.sidebar.title("🩺 医疗问答 Agent")
user_question = st.sidebar.text_input("请输入问题", "")

# ------------------------------
# 执行工作流
# ------------------------------
if user_question:
    with st.spinner("正在检索知识并生成推理链..."):
        result_dict = agent.run(user_question)

        # 格式化结果，补全 final_answer
        # 如果置信度过低，可在前端提示用户人工复核
        final_answer = (
            "⚠️ 置信度低，请人工复核。" 
            if result_dict["confidence"] < agent.config["threshold"] 
            else "✅ 自动回答结果"
        )

        formatted_result = format_agent_result(
            query=result_dict["question"],
            retrieved_docs=result_dict.get("retriever_docs", []),
            reasoning_chain=result_dict["reasoning"],
            verify_result=result_dict["verify_result"],
            confidence=result_dict["confidence"],
            final_answer=final_answer
        )

        # ------------------------------
        # 展示 UI
        # ------------------------------
        render_streamlit_ui(formatted_result)
