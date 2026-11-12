# The Relay — Research Paper Assistant

A Python research-paper question-answering system that retrieves evidence from three arXiv papers and generates grounded, cited answers using Groq.

## Features

- Retrieves passages from three specified research papers.
- TF-IDF retrieval with configurable top-k passages.
- Experimental hybrid retrieval using FastEmbed embeddings and reciprocal-rank fusion.
- Groq streaming inference using `openai/gpt-oss-20b`.
- Source-grounded answering and unsupported-question refusal.
- Configurable asynchronous stand-in for latency, streaming, rate limits, failures, timeouts and overload.
- Explicit retry handling and request metrics.

## Project structure

```text
the-relay/
├── papers/                 # Research paper PDFs
├── data/                   # Extracted chunks
├── src/
│   ├── retriever.py
│   ├── hybrid_retriever.py
│   ├── model_client.py
│   ├── pipeline.py
│   └── standin.py
├── tests/                  # Stand-in and retry tests
├── experiments/            # Raw results and evaluations
├── reports/                # Assignment report
├── download_papers.py
├── extract_chunks.py
├── requirements.txt
└── README.md
```

## Setup

Python 3.14.5 was used during development.

```bash
git clone https://github.com/Medha170/Relay-Assignment
cd the-relay

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit `.env` or expose your API key.

## Prepare the paper corpus

Download the papers and extract their text chunks:

```bash
python download_papers.py
python extract_chunks.py
```

The extracted chunks are stored in `data/chunks.json`.

## Run the assistant

Run using the real Groq service:

```bash
python -m src.pipeline
```

Run using the stand-in:

```bash
MODEL_BACKEND=standin python -m src.pipeline
```

The stand-in allows pipeline behaviour to be tested without making real model requests.

## Tests

Run the retry tests:

```bash
python -m tests.test_retries
```

## Experiments

The `experiments/` directory contains raw request measurements, retrieval comparisons and evaluation scripts.

The combined request log is:

```text
experiments/results.csv
```

It records request source, scenario, latency, status, retries, token usage, retrieval setting and manual answer grade.

## Limitations

- The stand-in generates canned responses and does not evaluate answer quality.
- Real API limits beyond the observed TPM limit were not independently measured.
- Retrieval evaluation measures paper-level ranking, not full passage relevance.
- Answer-generation quality was not compared across all retrieval configurations.

## Technologies

Python, FastAPI-compatible asynchronous patterns, Groq API, scikit-learn, FastEmbed, NumPy, pandas and PyMuPDF.