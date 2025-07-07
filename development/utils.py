# utils.py

import csv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
import unicodedata


# 1️⃣ Load CSV as Documents for LangChain


def strip_accents(s: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def load_csv_documents(csv_path: str) -> list[Document]:
    clean_rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k: strip_accents(v) for k, v in row.items()}
            clean_rows.append(clean_row)

    docs = []
    for row in clean_rows:
        content = "\n".join(f"{k}: {row[k]}" for k in [
                            "name", "background", "info"])
        metadata = {k: row[k] for k in ["id", "office", "position"]}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs
