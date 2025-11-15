# modules/verifier.py
import re
from sentence_transformers import SentenceTransformer, util

class Verifier:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        校验模块：
        - 检查逻辑连贯性
        - 检查医学术语合理性
        - 输出校验得分 (0~1)
        """
        print(f"[Verifier] Loading verification model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        # 医学常用术语词表，可自行扩充
        self.medical_keywords = [
            "疾病", "症状", "治疗", "诊断", "药物", "病因", "并发症",
            "心脏", "肝脏", "肾脏", "糖尿病", "血压", "感染", "炎症"
        ]

    def check_medical_terms(self, text):
        """
        检查是否出现了医学关键词
        """
        hits = [w for w in self.medical_keywords if w in text]
        term_score = len(hits) / len(self.medical_keywords)
        return min(term_score * 2, 1.0)  # 强化医学相关性得分

    def check_logical_connectors(self, text):
        """
        检查逻辑词汇（推理合理性）
        """
        connectors = ["因此", "所以", "导致", "因为", "说明", "结果", "由此"]
        hits = [c for c in connectors if c in text]
        return min(len(hits) / 5, 1.0)

    def semantic_consistency(self, question, reasoning_text):
        """
        计算语义一致性（问题与推理链的相关性）
        """
        q_emb = self.embedding_model.encode(question, convert_to_tensor=True)
        r_emb = self.embedding_model.encode(reasoning_text, convert_to_tensor=True)
        similarity = util.cos_sim(q_emb, r_emb).item()
        return float(similarity)

    def verify(self, question, reasoning_text):
        """
        综合打分模块：
        - 医学相关度
        - 逻辑连贯度
        - 语义一致性
        """
        term_score = self.check_medical_terms(reasoning_text)
        logic_score = self.check_logical_connectors(reasoning_text)
        semantic_score = self.semantic_consistency(question, reasoning_text)

        total_score = round(0.3 * term_score + 0.3 * logic_score + 0.4 * semantic_score, 3)
        result = {
            "term_score": term_score,
            "logic_score": logic_score,
            "semantic_score": semantic_score,
            "verify_score": total_score
        }

        print(f"[Verifier] ✅ Verification completed: {result}")
        return result
