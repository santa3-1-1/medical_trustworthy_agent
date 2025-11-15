# main.py
from modules.workflow import MedicalTrustworthyAgent
from utils.visualization import format_agent_result

if __name__ == "__main__":
    print("============================================")
    print("🏥 Medical Trustworthy Agent (English Mode)")
    print("============================================")
    print("Your knowledge base is currently in English (PubMedQA).")
    print("Please enter your medical question in English.\n")

    agent = MedicalTrustworthyAgent()

    question = input("📝 Enter your medical question: ")

    result = agent.run(question)

    print("\n============== FINAL OUTPUT ==============")
    print(result)
