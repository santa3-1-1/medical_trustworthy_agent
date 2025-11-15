# prepare_knowledge_base.py
# Author: chen 项目专用版本

from datasets import load_dataset
import json
import os

# =====================
#  1. 切换数据规模
# =====================
MODE = "small"   # small | medium | full

OUTPUT_DIR = "knowledge_dir"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[Info] Current MODE = {MODE}")

# =====================
#  2. 定义不同规模的数据集
# =====================
DATASETS = {
    "small": [
        ("pubmed_qa", "pqa_labeled"),   # 很小，适合本地 CPU
    ],
    "medium": [
        ("pubmed_qa", "pqa_labeled"),
        ("medmcqa", "dev"),
    ],
    "full": [
        ("pubmed_qa", "pqa_labeled"),
        ("medmcqa", "train"),
        ("medqa", "usmle"),
    ],
}

# =====================
# 3. 统一转换成你的 RAG 文档格式
# =====================
def convert_pubmed_qa(example):
    return {
        "question": example.get("question", ""),
        "answer": example.get("final_decision", ""),
        "context": example.get("context", ""),
    }

def convert_medmcqa(example):
    return {
        "question": example.get("question", ""),
        "answer": example.get("cop", ""),
        "context": " | ".join(example.get("options", [])),
    }

def convert_medqa(example):
    return {
        "question": example.get("question", ""),
        "answer": example.get("answer", ""),
        "context": example.get("long_answer", ""),
    }

CONVERTERS = {
    "pubmed_qa": convert_pubmed_qa,
    "medmcqa": convert_medmcqa,
    "medqa": convert_medqa,
}

# =====================
# 4. 下载并写入 knowledge_dir
# =====================
total_count = 0

for name, subset in DATASETS[MODE]:
    print(f"\n[Info] Loading dataset: {name} ({subset}) ...")
    ds = load_dataset(name, subset, split="train")

    converter = CONVERTERS[name]

    output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            converted = converter(item)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            total_count += 1

    print(f"[Done] Saved to {output_path} ({len(ds)} items)")

print(f"\n[Success] Total documents loaded: {total_count}")
print(f"[Success] Knowledge base prepared in {OUTPUT_DIR}/")
