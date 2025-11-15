# utils/visualization.py

def format_agent_result(query, retrieved_docs, reasoning_chain, verify_result, confidence, final_answer):
    """
    标准化输出格式，让 UI 层（Streamlit / Gradio）复用。
    """
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "reasoning_chain": reasoning_chain,
        "verify_result": verify_result,
        "confidence": confidence,
        "final_answer": final_answer
    }
