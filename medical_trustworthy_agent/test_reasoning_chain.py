# test_reasoning_chain.py
# 放在 medical_trustworthy_agent/ 目录下
# 用于测试 modules/reasoning_chain.py 是否正常生成英文推理链

from modules.reasoning_chain import ReasoningChain

def main():
    # 初始化推理链模块
    rc = ReasoningChain(model_name="facebook/opt-350m", max_new_tokens=256)

    # 测试问题
    question = "What are the common treatments for hypertension?"

    # 模拟检索到的文献片段
    retrieved_docs = [
        {"text": "Hypertension is often treated with lifestyle modifications, such as reducing salt intake, exercising regularly, and maintaining a healthy weight."},
        {"text": "Medications for hypertension include ACE inhibitors, beta-blockers, diuretics, and calcium channel blockers."},
        {"text": "Patient education and regular monitoring of blood pressure are important components of managing hypertension."}
    ]

    # 生成推理链
    reasoning = rc.generate(question, retrieved_docs)

    print("\n=== TEST RESULT: Generated Reasoning Chain ===")
    print(reasoning)
    print("=== END OF TEST ===\n")

if __name__ == "__main__":
    main()
