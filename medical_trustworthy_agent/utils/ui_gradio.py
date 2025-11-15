# utils/ui_gradio.py
import gradio as gr

def render_gradio_interface(run_agent_fn):

    def _call_agent(query):
        result = run_agent_fn(query)

        return (
            result["retrieved_docs"],
            result["reasoning_chain"],
            result["verify_result"],
            result["confidence"],
            result["final_answer"]
        )

    demo = gr.Interface(
        fn=_call_agent,
        inputs=gr.Textbox(lines=2, label="请输入医学问题"),
        outputs=[
            gr.JSON(label="检索文档"),
            gr.Markdown(label="推理链"),
            gr.JSON(label="校验结果"),
            gr.Number(label="置信度"),
            gr.Markdown(label="最终答案"),
        ],
        title="🩺 医疗问答可信 Agent"
    )
    return demo
