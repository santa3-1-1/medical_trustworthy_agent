class ConfidenceCalculator:
    """
    结合 LLM 内部置信度与校验模块得分，输出综合置信度
    """

    def __init__(self, alpha=0.6):
        self.alpha = alpha

    def calculate(self, llm_score: float, verify_score: float) -> float:
        confidence = round(self.alpha * llm_score + (1 - self.alpha) * verify_score, 2)
        return confidence

# 测试（可选）
if __name__ == "__main__":
    estimator = ConfidenceCalculator(alpha=0.6)
    print("综合置信度:", estimator.calculate(0.8, 0.7))
