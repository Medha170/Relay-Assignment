# The Relay — Research Paper Assistant

## 1. What I built and why

I built a Python research-paper assistant that retrieves passages from three specified arXiv papers and uses Groq (`openai/gpt-oss-20b`) to answer questions with source citations. The pipeline performs retrieval, checks whether passages are available, constructs a grounded prompt, and streams the model response. It refuses questions unsupported by the papers.

The corpus contains 384 chunks: 202 Moshi, 142 Dynamic Chunking, and 40 Attention Is All You Need. I used 220-word chunks with 40-word overlap and TF-IDF retrieval, fetching the top three passages in the answer pipeline.

## 2. The real service and my stand-in

The real service is Groq's hosted API. My stand-in is a configurable asynchronous Python service that simulates latency, streaming, request-rate limits, failures, timeouts, and overload delays. It returns canned text, not actual model answers, so it tests service behaviour rather than answer quality.

I disabled SDK retries and implemented up to two retries for rate-limit errors, respecting `Retry-After` when available. I do not retry after partial streamed output, avoiding duplicated answers.

Groq's documented limits include RPM, RPD, TPM and TPD. My account returned an 8,000 TPM limit in actual 429 responses. I did not independently measure the other limits.

**Table 1 — Real service limits**

| Limit / behaviour | Finding | Confidence |
|---|---|---|
| Requests/minute | Not measured | — |
| Tokens/minute | 8,000 TPM | High; actual 429 |
| Requests/day | Not measured | — |
| Tokens/day | Not measured | — |
| Requests in flight | Not measured | — |
| Limit response | HTTP 429, `rate_limit_exceeded`, token limit details and `Retry-After` | High |
| Observed retry-after | Approximately 0.165–9.818 seconds | High; error responses |
| Recovery time | Not independently measured | — |

## 3. Results

I recorded 50 requests: 15 real Groq requests and 35 stand-in requests. Six real requests succeeded and nine were rate-limited. All 35 stand-in requests succeeded. Percentiles below use successful requests only; real TTFT was not recorded.

**Table 2a — Latency and offered load**

| Scenario | Source | Requests | Duration | Offered/min | TTFT p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
|---|---|---:|---:|---:|---|---|
| Quality run | Real | 15 | Not measured | Not measured | Not measured | 1067/1451/1510* |
| Steady | Stand-in | 5 | 1.96s | 153.18 | 303/305/305 | 391/393/393 |
| Burst | Stand-in | 10 | 0.89s | 671.45 | 801/802/802 | 885/889/889 |
| Sustained overload | Stand-in | 20 | 2.81s | 427.35 | 804/805/805 | 892/894/894 |
| Halved per-minute limit | Not run | — | Not measured | Not measured | Not measured | Not measured |
| Daily budget exhausted | Not run | — | Not measured | Not measured | Not measured | Not measured |

*Real latency percentiles are for the six successful requests only. The nine rate-limited responses are included in outcome counts, not latency percentiles. Stand-in offered rates are approximate, calculated from short concurrent batches; they are not sustained production rates.

**Table 2b — Outcomes**

| Scenario | Success | Rate-limited | Timeout/error | Retries/request | Peak queue | Tokens/answer | Cost/1,000 |
|---|---:|---:|---:|---|---|---|---|
| Real Groq | 40% | 60% | 0% | Not measured | Not measured | Not measured | Not measured |
| Stand-in steady | 100% | 0% | 0% | 0 | Not measured | Not measured | Not measured |
| Stand-in burst | 100% | 0% | 0% | 0 | Not measured | Not measured | Not measured |
| Stand-in overload | 100% | 0% | 0% | 0 | Not measured | Not measured | Not measured |

The stand-in's burst and overload runs were slower than steady traffic because of configured concurrency and overload delays. It did not reproduce Groq's token-based rate limit. Sample sizes are small, so tail percentiles should not be treated as stable benchmarks.

I wrote 15 quality questions. In the original real run, six completed and nine were rate-limited. The current manual grades are three correct, two wrong, and one correctly refused. One answer (Q8) was truncated in the original run but was later independently retested successfully; its grade reflects that later retest, not the original output. The stand-in cannot be graded for answer quality.

## 4. Where time goes

The real successful requests had a total-latency p50 of about 1.07 seconds. TTFT was not captured, so I cannot separate retrieval, queueing, model startup and generation time reliably. The stand-in exposes the effect of configured initial delay and overload delay, but does not model Groq's internal execution.

## 5. When limit hit

Nine of fifteen real requests received 429 responses. The error payloads identified the token-per-minute limit of 8,000. I stopped rather than repeatedly retrying or deliberately exhausting the quota. The retry tests used mocked 429 responses, not additional real quota consumption.

## Retrieval: quality versus cost

I compared TF-IDF with a hybrid retriever using FastEmbed (`BAAI/bge-small-en-v1.5`) and reciprocal-rank fusion. On 12 in-scope questions, both methods found the expected paper in the top three (12/12); paper-level MRR was 1.0 for both. This does not prove passage-level relevance or answer correctness.

**Table 3 — Retrieval settings**

| Setting | Chunk size | Fetched | Rerank | Questions | Paper Hit@3 | Answer quality / tokens |
|---|---:|---:|---|---:|---|---|
| TF-IDF pipeline | 220 words | 3 | No | 15 | 12/12 in-scope | Not measured by setting |
| TF-IDF evaluation retrieval | 220 words | 5 | No | 15 | Not separately measured | Not measured |
| Hybrid comparison | 220 words | 3 | Yes | 15 | 12/12 in-scope | Not measured by setting |

## How this library will mislead a naive retriever (3 examples)

1. A Bengaluru population question retrieved unrelated passages about causal masking and learning-rate adjustment.
2. An Olympic 100m question retrieved Moshi dialogue examples and unrelated chunking material.
3. A dot-product scaling question retrieved the correct Attention passage, but also a Dynamic Chunking passage discussing inverse-square-root learning-rate scaling.

Similarity is not answer confidence. A score threshold of 0.10 would reject a relevant Q12 passage (0.0747) while accepting the unrelated Bengaluru result (0.1291). I therefore removed the threshold and relied on evidence-grounded model refusal instead.

## Your feedback on Vaani

What stood out was the developer workflow: an application triggers a call, the agent can use tools during the conversation, and the outcome returns through webhooks. The separation between dispatch acceptance and actual call completion is useful to understand. I found the overall product story easier to follow than the details of how the voice stack behaves internally. I would add a small end-to-end developer example that shows a real request, a tool call, and the final webhook payload together.

## 6. Retries

The client allows two retries after the initial attempt, uses `Retry-After` when supplied, and otherwise uses exponential backoff. SDK retries are disabled to keep retry behaviour explicit. Tests verified success after one mocked 429 and propagation after three consecutive 429s. Partial streams are not retried.

## 7. What the person sees

The user receives streamed answers with source references when evidence is available. Unsupported questions should receive: “The provided papers do not contain enough information to answer this question.” Retrieval can still return irrelevant passages, so the refusal behaviour is important.

## 8. What I'm unsure about, what I left out, what I learned

I did not measure real TTFT, token usage, daily limits, concurrency limits, queue length, or cost. I did not run the halved-limit or exhausted-daily-budget scenarios. The stand-in does not reproduce actual model quality or Groq's TPM accounting. I learned that retrieval relevance, answer correctness, and service reliability are separate problems, and each needs its own measurements.

## How I worked with the assistant

I used the assistant for incremental debugging, test design, interpreting experiment outputs, and report structure. Three important prompts, reconstructed from the work, were: (1) help debug the asynchronous Groq streaming client without rewriting the whole project; (2) help design stand-in tests for latency, failures, rate limits and concurrency; (3) help interpret raw experiment results without inventing missing measurements.

One assistant suggestion I rejected was using a fixed relevance-score threshold. The Bengaluru example showed that an unrelated passage could score higher than a relevant one. I also chose to keep TF-IDF in the actual pipeline rather than switch to hybrid retrieval, because the evaluation did not demonstrate a meaningful paper-level improvement.

The implementation, paper corpus, experiment execution, grading decisions, interpretation of the retrieval failures, and final design choices were my work. The assistant helped me reason through problems and review the evidence.