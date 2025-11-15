# utils/ui_streamlit.py
import streamlit as st

def render_streamlit_ui(result):
    st.title("🩺 医疗问答可信Agent")

    st.markdown("### 💬 输入问题")
    st.info(result["query"])

    st.markdown("### 📚 检索到的医学文献")
    docs_container = st.container()
    with docs_container:
        for i, doc in enumerate(result["retrieved_docs"]):
            st.write(f"**文档 {i+1}:**")
            st.write(doc[:500] + "..." if len(doc) > 500 else doc)
            st.markdown("---")  # 分隔文档

    st.markdown("### 🧠 推理链 (Reasoning Chain)")
    st.code(result["reasoning_chain"], language="markdown")

    st.markdown("### 🧩 校验结果 (Verification)")
    st.json(result["verify_result"])

    st.markdown("### 📈 置信度 (Confidence)")
    # 用列显示 metric，避免在循环里触发节点冲突
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("综合置信度", f"{result['confidence']:.3f}")

    st.markdown("### ✅ 最终输出")
    st.success(result["final_answer"])
