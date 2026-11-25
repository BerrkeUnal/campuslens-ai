# Model Warm-up and GPU Reranking Validation

## 1. Scope

This work introduces explicit model preparation, cache-only runtime
loading, background model warm-up, question endpoint readiness
protection, and optional CUDA acceleration for the BGE reranker.

CPU remains the portable default. Explicit CUDA configuration does
not silently fall back to CPU.

The public question response schema and strict source validation
contract are preserved.

## 2. Experimental environment

| Component | Configuration |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.16 |
| CPU PyTorch | 2.15.0+cpu |
| GPU PyTorch | 2.15.0+cu132 |
| CUDA runtime | 13.2 |
| GPU | NVIDIA GeForce RTX 4060, approximately 8 GiB |
| Embedding | intfloat/multilingual-e5-small |
| Sparse retrieval | Qdrant/bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Reranker precision | FP32 |
| LLM | Ollama qwen3:4b |
| Ollama version | 0.40.18 |
| Observed LLM context | 4096 tokens |
| Qdrant | Embedded local mode |

CPU and GPU Python environments were isolated. The GPU environment
was cloned from the CPU environment and its PyTorch build was replaced
with the CUDA-enabled build. The original CPU environment was retained.

## 3. Model preparation and warm-up

Model preparation is explicit. Runtime model loading uses cached
model files rather than initiating implicit downloads.

Three independent fresh-process benchmark runs were performed.
The disk cache was retained.

| Component | Cold median (ms) | Warm median (ms) | Speedup |
|---|---:|---:|---:|
| Embedding | 2594.457 | 11.634 | 223.01x |
| Reranker | 4312.904 | 108.312 | 39.82x |
| Ollama | 3853.156 | 187.779 | 20.52x |

These measurements represent component-level technical warm-up,
not complete RAG requests. Cold means non-resident model state,
not an empty disk cache.

## 4. Controlled CPU/GPU reranking benchmark

Six questions were evaluated using the same 20 retrieval candidates
per question. Each device performed three measured repetitions.

| Metric | CPU | GPU |
|---|---:|---:|
| Timed predictions | 18 | 18 |
| Timed question-chunk pairs | 360 | 360 |
| Mean latency (ms) | 17449.147 | 813.899 |
| Median latency (ms) | 17510.020 | 808.176 |
| Exact chunk Hit@5 | 1.000000 | 1.000000 |
| Exact chunk MRR@5 | 0.916667 | 0.916667 |

Mean speedup: 21.44x.

Mean latency reduction: 95.34%.

All 18 paired predictions preserved the complete candidate ranking.
All 360 paired scores were within the declared numerical tolerance
(rtol=1e-4, atol=1e-5).

Maximum absolute score difference: approximately 1.52e-6.

This experiment used direct CrossEncoder inference and did not
measure complete production request latency.

## 5. Production CPU/GPU validation

The updated production Reranker implementation was tested using
the same six questions and 20 candidates per question.

| Metric | CPU | GPU |
|---|---:|---:|
| Timed predictions | 6 | 6 |
| Timed question-chunk pairs | 120 | 120 |
| Mean reranking latency (ms) | 14352.778 | 793.226 |
| Median reranking latency (ms) | 13615.874 | 794.259 |
| Exact chunk Hit@5 | 1.000000 | 1.000000 |
| Exact chunk MRR@5 | 0.916667 | 0.916667 |
| Historical ranking agreement | 6/6 | 6/6 |
| Maximum score difference from respective baseline | 0 | 0 |

Observed production GPU speedup: 18.09x.

Observed production reranking latency reduction: 94.47%.

Both devices preserved their respective historical ranking and
score results. These measurements use one repetition per question
and should not be interpreted as universal performance guarantees.

### Production GPU startup

The actual FastAPI lifespan reached READY with all three components.

| Component | Duration (ms) |
|---|---:|
| Embedding | 3228.837 |
| Reranker | 5375.411 |
| Ollama | 3887.988 |
| Sum of component durations | 12492.236 |

The BGE model parameters were on cuda:0 with FP32 precision.
Ollama reported qwen3:4b at 100% GPU placement.

Both models remained resident during sequential reranking and
technical LLM inference.

Minimum sampled free memory:

- System GPU measurement: 1059 MiB.
- CUDA runtime measurement: 692.258 MiB.

Peak PyTorch allocated memory: 2540.352 MiB.
Peak PyTorch reserved memory: 3192 MiB.

The two free-memory measurements are reported separately.
They are not assumed to be interchangeable.

Concurrent inference and maximum-length LLM generation were not
validated by this experiment.

## 6. Real retrieval validation

The local validation collection contained one official university
regulation, covering 14 pages and 27 indexed chunks.

Six matching answerable questions were evaluated using four
retrieval methods, for 24 real retrieval operations.

| Method | Exact chunk Hit@5 | Exact chunk MRR@5 | Mean latency (ms) |
|---|---:|---:|---:|
| BM25 | 1.000000 | 0.888889 | 3.698 |
| Dense | 1.000000 | 0.680556 | 19.364 |
| Hybrid | 1.000000 | 0.791667 | 21.531 |
| Hybrid + Rerank (CPU) | 1.000000 | 0.916667 | 17179.647 |

This is a single-document subset experiment, not the full
50-question evaluation dataset.

## 7. End-to-end RAG validation

Eight requests were executed through the real question API using
embedded Qdrant, hybrid retrieval, GPU reranking, local Ollama,
and strict source validation.

No model or retrieval mocks were used.

| Metric | Few-shot baseline | Zero-shot experiment |
|---|---:|---:|
| Executed requests | 8 | 8 |
| HTTP 200 | 8 | 3 |
| HTTP 503 | 0 | 5 |
| Correct core answer among answerable cases | 2/6 | 0/6 |
| Gold source present in successful API responses | 5/6 | 1/6 |
| False abstention | 1 | 1 |
| Correct out-of-scope abstention | 2/2 | 2/2 |
| Mean successful request latency (ms) | 2511.723 | 2311.523 |

The latency means are not directly comparable because the
zero-shot experiment had five failed requests.

### Manual baseline answer assessment

| Case | Expected core answer | Observed result |
|---|---|---|
| eval-001 | No make-up exam after attendance failure | Incorrect |
| eval-002 | Three working days | Correct core, incomplete |
| eval-003 | No single-course exam for internship | Incorrect |
| eval-004 | Most recently received grade | Unrelated answer |
| eval-005 | Five working days | False abstention |
| eval-006 | GPA at least 3.50, subject to conditions | Correct core, incomplete |
| local-oos-001 | Insufficient context | Correct abstention |
| local-oos-002 | Insufficient context | Correct abstention |

HTTP success and valid source identifiers do not establish
semantic answer correctness.

### Rejected zero-shot experiment

The few-shot examples were removed and explicit grounding
instructions were added. All 77 scoped unit tests passed.

However, five real answerable requests failed strict citation
validation because inline citations did not match used_source_ids.

The remaining answerable request repeated the question and
incorrectly reported insufficient context.

The zero-shot change was therefore reverted. Its results and
patch were preserved as experimental evidence.

Rejected raw LLM answers were not captured, so their semantic
correctness cannot be determined.

The original source validation contract was not weakened.

## 8. Known limitations and follow-up

- qwen3:4b does not provide sufficiently reliable semantic answers
  on this small validation set.
- Source identifier validation does not prove that a cited passage
  semantically supports a generated claim.
- Model readiness establishes technical availability, not answer
  correctness.
- Sequential GPU coexistence does not guarantee concurrent inference.
- Full-dataset answer correctness was not measured.
- The single-document subset is not a representative held-out test.
- The baseline embedded Qdrant experiment emitted a destructor
  warning during Python shutdown. Explicit cleanup succeeded in
  the subsequent experiment script, but production lifecycle
  cleanup was not changed.
- The existing health endpoint does not expose model readiness.
- The evaluation endpoint does not currently share the question
  endpoint readiness guard; this requires coordination with the
  evaluation workflow owner.

## 9. Evidence locations

Raw evidence is retained in ignored local data directories.
These directories are not automatically included in the PR.

| Experiment | Evidence directory |
|---|---|
| Warm-up benchmark | data/benchmarks/model_warmup_20261125_173538 |
| Real retrieval | data/e2e_validation/model_warmup_readiness/results/retrieval_20261125_184906 |
| Controlled CPU/GPU | data/e2e_validation/model_warmup_readiness/results/bge_device_benchmark_20261125_194454 |
| GPU coexistence | data/e2e_validation/model_warmup_readiness/results/ollama_bge_coexistence_20261125_201059 |
| Production CPU | data/e2e_validation/model_warmup_readiness/results/production_cpu_20261125_210324 |
| Production GPU | data/e2e_validation/model_warmup_readiness/results/production_gpu_20261125_210909 |
| RAG baseline | data/e2e_validation/model_warmup_readiness/results/production_rag_20261125_211205 |
| Rejected zero-shot | data/e2e_validation/model_warmup_readiness/results/production_rag_zero_shot_20261125_211816 |

JSON results, execution logs and checksum manifests are retained
in the corresponding experiment directories.

## 10. Conclusion

The model preparation, readiness and optional CUDA reranking
implementation was validated in real production execution.

GPU reranking substantially reduced measured latency while
preserving candidate ranking on the tested subset.

The local LLM's semantic answer quality remains a separate,
unresolved limitation. This work does not claim that the complete
RAG system has achieved production-grade answer accuracy.
