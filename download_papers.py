import requests
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need": "https://arxiv.org/pdf/1706.03762v7",
    "moshi": "https://arxiv.org/pdf/2410.00037v1",
    "dynamic_chunking": "https://arxiv.org/pdf/2507.07955v2",
}

Path("papers").mkdir(exist_ok=True)

for name, url in PAPERS.items():
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    path = Path("papers") / f"{name}.pdf"
    path.write_bytes(response.content)

    print(f"Downloaded {name}: {path.stat().st_size:,} bytes")