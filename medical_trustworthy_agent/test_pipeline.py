# test_pipeline.py
from modules.workflow import MedicalTrustworthyAgent

def test_pipeline(questions):
    """
    批量测试 MedicalTrustworthyAgent 全流程。
    输出每一步日志，便于验证流程。
    """
    agent = MedicalTrustworthyAgent()

    for idx, question in enumerate(questions, 1):
        print("\n" + "="*50)
        print(f"🔹 测试问题 {idx}: {question}")
        print("="*50)

        result = agent.run(question)

        print("\n📄 完整输出结果：")
        print(result)
        print("\n" + "-"*50)

if __name__ == "__main__":
    # 示例问题列表
    questions_to_test = [
        "What are the common treatments for hypertension?",
        "How is diabetes managed in adults?",
        "What are the side effects of ACE inhibitors?",
    ]

    test_pipeline(questions_to_test)
