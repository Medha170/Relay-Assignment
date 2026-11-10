import json
import fitz
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need": {
        "file": "papers/attention_is_all_you_need.pdf",
        "title": "Attention Is All You Need",
        "arxiv": "1706.03762v7",
    },
    "moshi": {
        "file": "papers/moshi.pdf",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv": "2410.00037v1",
    },
    "dynamic_chunking": {
        "file": "papers/dynamic_chunking.pdf",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv": "2507.07955v2",
    },
}

CHUNK_SIZE = 220
OVERLAP = 40

def split_into_chunks(text):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


all_chunks = []

for paper_id, details in PAPERS.items():
    pdf = fitz.open(details["file"])

    for page_index, page in enumerate(pdf):
        text = page.get_text("text")

        for chunk_index, chunk in enumerate(split_into_chunks(text)):
            all_chunks.append({
                "paper_id": paper_id,
                "title": details["title"],
                "arxiv": details["arxiv"],
                "page": page_index + 1,
                "chunk_id": chunk_index,
                "text": chunk,
            })

    print(f"{details['title']}: {len(pdf)} pages processed")
    pdf.close()

Path("data").mkdir(exist_ok=True)

with open("data/chunks.json", "w", encoding="utf-8") as file:
    json.dump(all_chunks, file, indent=2, ensure_ascii=False)

print(f"\nTotal chunks created: {len(all_chunks)}")