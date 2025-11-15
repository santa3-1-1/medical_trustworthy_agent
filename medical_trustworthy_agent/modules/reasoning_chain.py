# modules/reasoning_chain.py
from transformers import pipeline

class ReasoningChain:
    def __init__(self, model_name="facebook/opt-350m", max_new_tokens=256):
        """
        使用指定模型生成医学推理链。
        """
        print(f"[ReasoningChain] Loading reasoning model: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device_map=None,  # CPU
            torch_dtype=None,  # 避免 auto 在 CPU 上出错
            max_new_tokens=max_new_tokens
        )

    def build_prompt(self, question, retrieved_docs):
        """
        构造 prompt：英文问题 + 文献摘要
        优化 prompt，去掉多余的开头重复句
        """
        # 截取每篇文档前 500 字
        context = "\n\n".join([doc["text"][:500] for doc in retrieved_docs])

        prompt = f"""
    You are a professional medical reasoning assistant.
    Using the following medical knowledge, answer the question and generate a reasoning chain.

    Question:
    {question}

    Medical knowledge:
    {context}

    ### Start generating reasoning chain and conclusion from here
    """
        return prompt

    def generate(self, question, retrieved_docs):
        if not retrieved_docs:
            return "⚠️ No documents retrieved, cannot generate reasoning chain."

        prompt = self.build_prompt(question, retrieved_docs)
        print(f"[ReasoningChain] Generating reasoning chain with {self.generator.model.name_or_path}...")

        output = self.generator(
            prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )[0]["generated_text"]

        # 去掉 prompt 部分，只保留模型生成的内容
        generated_text = output[len(prompt):].strip()
        return generated_text
