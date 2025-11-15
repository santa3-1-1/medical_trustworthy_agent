# modules/workflow.py

import os
import yaml
from modules.retriever import Retriever
from modules.reasoning_chain import ReasoningChain
from modules.verifier import Verifier
from modules.confidence import ConfidenceCalculator
from utils.embeddings import EmbeddingModel

class MedicalTrustworthyAgent:
    def __init__(self, config_path="config.yaml"):

        default_config = {
            "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
            "reasoner_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "alpha": 0.6,
            "threshold": 0.7,
            "knowledge_dir": "knowledge_base",
            "top_k": 3
        }

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            self.config = {**default_config, **file_config}
            print(f"[Agent] Loaded config from {config_path}")
        else:
            self.config = default_config
            print(f"[Agent] Using default config")

        embedding_model = EmbeddingModel(model_name=self.config["retriever_model"])

        self.retriever = Retriever(
            knowledge_dir=self.config["knowledge_dir"],
            embedding_model=embedding_model,
            top_k=self.config.get("top_k", 3)
        )

        self.reasoner = ReasoningChain(model_name=self.config["reasoner_model"])
        self.verifier = Verifier()
        self.confidence_calc = ConfidenceCalculator(alpha=self.config["alpha"])

    def run(self, question):
        docs = self.retriever.retrieve(question)
        reasoning = self.reasoner.generate(question, docs)
        verify_result = self.verifier.verify(question, reasoning)

        llm_score = 0.85
        confidence = self.confidence_calc.calculate(
            llm_score, verify_result["verify_score"]
        )

        # 你可以换成 LLM 最终 answer，这里先用推理链最后一句
        final_answer = reasoning.split("\n")[-1].strip()

        return {
            "query": question,
            "retrieved_docs": docs,
            "reasoning_chain": reasoning,
            "verify_result": verify_result,
            "confidence": confidence,
            "final_answer": final_answer
        }
