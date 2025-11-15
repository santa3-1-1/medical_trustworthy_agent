# utils/data_loader.py
import os
import json
from typing import List
from langchain.docstore.document import Document
from PyPDF2 import PdfReader

class DataLoader:
    def __init__(self, knowledge_dir="knowledge_dir"):
        """
        医学知识库加载器：只需修改路径
        - 支持 txt / pdf / json
        - 输出 Document 对象用于向量化
        """
        self.knowledge_dir = knowledge_dir
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)

    def load_text_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": path})]

    def load_json_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = []
        for entry in data:
            content = entry.get("content") or entry.get("text") or ""
            docs.append(Document(page_content=content, metadata={"source": path}))
        return docs

    def load_pdf_file(self, path):
        reader = PdfReader(path)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                docs.append(Document(page_content=text, metadata={"source": f"{path}_page_{i}"}))
        return docs

    def load_jsonl_file(self, path):
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                content = data.get("text") or data.get("answer") or data.get("context") or ""
                docs.append(Document(page_content=content, metadata={"source": path}))
        return docs


    def load_all(self) -> List[Document]:
        """
        遍历目录加载所有支持的文件
        """
        print(f"[DataLoader] Loading knowledge from {self.knowledge_dir} ...")
        all_docs = []
        for root, _, files in os.walk(self.knowledge_dir):
            for file in files:
                path = os.path.join(root, file)
                if file.endswith(".txt"):
                    all_docs.extend(self.load_text_file(path))
                elif file.endswith(".pdf"):
                    all_docs.extend(self.load_pdf_file(path))
                elif file.endswith(".json"):
                    all_docs.extend(self.load_json_file(path))
                elif file.endswith(".jsonl"):
                    all_docs.extend(self.load_jsonl_file(path))

        print(f"[DataLoader] Total documents loaded: {len(all_docs)}")
        return all_docs

