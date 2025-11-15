# utils/ui_streamlit.py
import streamlit as st

def render_streamlit_ui(result):
    st.title("🩺 医疗问答可信Agent")

    st.markdown("### 💬 输入问题")
    st.info(result["query"])

    st.markdown("### 📚 检索到的医学文献")
    for i, doc in enumerate(result["retrieved_docs"]):
        st.write(f"**文档 {i+1}:**")
        st.write(doc[:500] + "..." if len(doc) > 500 else doc)

    st.markdown("### 🧠 推理链 (Reasoning Chain)")
    st.code(result["reasoning_chain"], language="markdown")

    st.markdown("### 🧩 校验结果 (Verification)")
    st.json(result["verify_result"])

    st.markdown("### 📈 置信度 (Confidence)")
    st.metric("综合置信度", result["confidence"])

    st.markdown("### ✅ 最终输出")
    st.success(result["final_answer"])
