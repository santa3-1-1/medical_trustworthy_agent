# app.py
import streamlit as st
from modules.workflow import MedicalTrustworthyAgent
from utils.ui_streamlit import render_streamlit_ui

st.set_page_config(page_title="医疗可信问答 Agent", layout="wide")

st.title("🩺 医疗可信问答 Agent")

agent = MedicalTrustworthyAgent()

query = st.text_input("请输入你的医学问题：")

if st.button("运行 Agent") and query:
    result = agent.run(query)
    render_streamlit_ui(result)
