# The Relay — Research Paper Assistant

## 1. What I built and why

A Python assistant retrieves passages from three specified arXiv papers and uses Groq (`openai/gpt-oss-20b`) for streamed, cited answers. It refuses unsupported questions. The corpus has 384 chunks (220 words, 40 overlap): 202 Moshi, 142 Dynamic Chunking, 40 Attention. The pipeline uses TF-IDF top three.

## 2. The real service and my stand-in

Groq is the hosted service. My configurable asynchronous stand-in simulates latency, streaming, rate limits, failures, timeouts and overload using canned text.

SDK retries are disabled. I implemented two retries after the initial attempt, using `Retry-After` or exponential backoff. Partial streams are not retried.

**Table 1 — Real service limits**

| Limit | Finding | Confidence |
|---|---|---|
| RPM, RPD, TPD, concurrency | Not measured | — |
| TPM | 8,000 | High; actual 429 |
| Response | HTTP 429, `rate_limit_exceeded` | High |
| Retry-after | 0.165–9.818 s | High |
| Recovery | Not measured | — |

## 3. Results

The CSV contains 57 requests: 22 real and 35 stand-in. The original real run had 15 requests; V2 added seven with TTFT instrumentation and a 500-token output limit.

**Table 2a — Latency and load**

| Scenario | N | Duration | Req/min | TTFT p50/95/99 | Total p50/95/99 ms |
|---|---:|---:|---:|---:|---:|
| Original real | 15 | NM | NM | NM | 1067/1451/1510 |
| V2 real | 7 | NM | NM | 765/1521/1526* | 1102/1699/1702 |
| Steady | 5 | 1.96s | 153 | 303/305/305 | 391/393/393 |
| Burst | 10 | 0.89s | 671 | 801/802/802 | 885/889/889 |
| Overload | 20 | 2.81s | 427 | 804/805/805 | 892/894/894 |
| Halved limit / daily exhausted | — | NM | NM | NM | NM |

*V2 TTFT has six observations; one request had no visible first token. Original latency uses six successful requests. NM = not measured; stand-in rates are approximate short-batch rates.*

**Table 2b — Outcomes**

| Scenario | Success | 429 | Timeout/error | Retries/request | Queue | Tokens in/out | Cost/1k |
|---|---:|---:|---:|---:|---|---|---|
| Original real | 40% | 60% | 0% | NM | NM | NM | NM |
| V2 real | 100% API | 0% | 0% | NM | NM | 1068/333 avg. | NM |
| Stand-in | 100% | 0% | 0% | 0 | NM | NM | NM |

V2 answers: four correct, two incomplete/wrong, one correctly refused. Two reached the 500-token cap. Across 13 manually graded answers: seven correct, four wrong/incomplete, two correctly refused. Q8's grade uses a later successful retest, not its original truncated output.

## 4. Where time goes

Original successful requests had total-latency p50 of 1.07 s; TTFT was not recorded. V2 TTFT p50 was 765 ms and total p50 1.10 s. Retrieval, queueing and inference were not measured separately.

## 5. When limit hit

Nine of 15 original requests received 429 responses identifying an 8,000 TPM limit. I stopped rather than exhaust quota. Retry tests used mocked 429s. V2 had no 429s, but two answers were incomplete at the output cap.

## Retrieval: quality versus cost

I compared TF-IDF with FastEmbed (`BAAI/bge-small-en-v1.5`) and reciprocal-rank fusion (RRF). Both retrieved the expected paper in the top three for all 12 in-scope questions (paper-level MRR 1.0). This does not prove passage relevance or answer correctness.

**Table 3 — Retrieval settings**

| Setting | Chunk | Fetched | Rerank | Qs | Correct+cited | Wrong | Correct refusal | Tokens/answer | TTFT p50/95 | Total p50/95 |
|---|---:|---:|---|---:|---|---|---|---|---|---|
| TF-IDF pipeline | 220 | 3 | No | 15 | NM | NM | NM | NM | NM | NM |
| TF-IDF evaluation | 220 | 5 | No | 15 | NM | NM | NM | NM | NM | NM |
| Hybrid | 220 | 3 | RRF | 15 | NM | NM | NM | NM | NM | NM |

Answer generation used TF-IDF top three only; equivalent generation tests across settings were not run.

## How this library will mislead a naive retriever

A Bengaluru population query retrieved causal-masking passages; an Olympic 100m query retrieved unrelated Moshi dialogue; a dot-product query returned a correct Attention passage alongside an unrelated learning-rate passage. A 0.10 threshold rejected relevant Q12 (0.0747) but accepted the unrelated Bengaluru result (0.1291). I removed the threshold because similarity is not answer confidence.

## Your feedback on Vaani

The tool-using voice-agent workflow and webhook delivery stood out. The distinction between dispatch acceptance and call completion was useful. An end-to-end example showing a request, tool call and final webhook payload would improve clarity.

## 6. Retries

Two retries follow the initial attempt. Mock tests verified recovery after one 429 and stopping after three consecutive 429s. Partial streams are not retried.

## 7. What the person sees

Users receive streamed answers with source references. Unsupported questions receive: “The provided papers do not contain enough information to answer this question.” Irrelevant retrieval and token limits can still produce incomplete answers.

## 8. What I'm unsure about, what I left out, what I learned

RPM, daily limits, concurrency, queue length, cost and recovery time were not measured. Halved-limit and daily-exhaustion tests were not run. The stand-in does not reproduce model quality or Groq TPM accounting. I learned to measure retrieval relevance, answer correctness and service reliability separately.

## How I worked with the assistant

Three reconstructed prompts: debug the async Groq client without rewriting the project; design stand-in tests; interpret raw results without inventing measurements. I rejected a fixed relevance threshold after the Bengaluru example and kept TF-IDF because hybrid showed no paper-level improvement. I implemented the pipeline, ran experiments, graded answers and made the final design choices; the assistant helped debug and review evidence.