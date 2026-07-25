# Roadmap

## Phase 0 — Framework

- [x] CMake project
- [x] Core library and CLI
- [x] CPU backend abstraction
- [x] Runtime and scheduler skeleton
- [x] Smoke test
- [x] Linux GCC and Clang CI definition
- [x] Structured engine status
- [x] Benchmark harness

## Phase 1A — CPU reference runtime

- [x] F32 dtype metadata
- [x] Aligned tensor storage
- [x] Tensor views and strides
- [x] Bounds-checked indexing
- [x] Elementwise add and multiply
- [x] Rank-2 matrix multiplication
- [x] RMSNorm
- [x] SiLU
- [x] Softmax
- [x] Synthetic transformer-like vertical slice
- [x] Correctness tests
- [x] Oracle endpoint configuration on port 5150

## Phase 1B — Predictable memory and GGUF foundations

- [x] Monotonic aligned arena allocator
- [x] Marks, rewinds, and scoped scratch lifetimes
- [x] Allocation and peak-memory telemetry
- [x] Reusable scratch-buffer planner
- [x] GGUF v2/v3 header and metadata parser
- [x] GGUF tensor descriptor parsing
- [x] GGUF inspection CLI with JSON output
- [x] Synthetic GGUF fixture tests

## Phase 1C — Memory-mapped GGUF tensor access

- [x] Read-only mapped-file abstraction
- [x] GGML block-size and storage-size registry
- [x] Checked tensor payload byte-size calculation
- [x] Zero-copy tensor views into mapped model data
- [x] Name-based tensor registry
- [x] Tensor bounds, alignment, and overlap validation
- [x] Mapping and payload telemetry
- [x] Verified GGUF inspection and named tensor lookup
- [x] Synthetic corruption and lifecycle tests

## Phase 1C.1 — Practical metadata export

- [x] Metadata-only inspection mode
- [x] Complete typed JSON metadata export
- [x] Exact-key and prefix filtering
- [x] Compact metadata summaries in descriptor and verified JSON
- [x] Bounded array and long-string previews for terminal output
- [x] Metadata-export correctness tests

## Phase 1D — Tokenizer and model manifest

- [ ] Architecture and hyperparameter manifest extraction
- [ ] Tokenizer vocabulary, merges, scores, and token-type loading
- [ ] BOS, EOS, unknown, padding, and special-token IDs
- [ ] Deterministic encode and decode
- [ ] Byte fallback and unknown-token handling
- [ ] Tokenizer round-trip tests
- [ ] Tensor-name and dimension validation against the manifest

## Phase 1E — Reference execution primitives

- [ ] Embedding lookup
- [ ] RoPE reference kernel
- [ ] Causal masked attention reference kernel
- [ ] Hybrid SSM reference-state representation
- [ ] KV-cache reference representation
- [ ] Minimal sampler
- [ ] Tiny end-to-end model fixture

## Phase 2 — First real model family

- [ ] F16 and BF16 reference decoding
- [ ] Q5_K and Q6_K reference decoding
- [ ] Architecture execution for the first supported hybrid model family
- [ ] Quantization adapters separated from core tensor representation
- [ ] Optional MTP-head discovery and loading
- [ ] Real prompt ingestion and token generation

## Phase 3 — CUDA backend

- [ ] Device discovery and capability reporting
- [ ] Explicit host/device memory planner
- [ ] F32 parity kernels
- [ ] Quantized matmul kernels
- [ ] Fused normalization, RoPE, and activation paths
- [ ] Stream and event management
- [ ] Benchmark parity against CPU reference

## Phase 4 — Oracle differentiators

- [ ] Memory-budget-driven layer placement
- [ ] KV-cache policy interface
- [ ] Quantized and tiered KV storage
- [ ] Context pressure telemetry
- [ ] Stable single-request low-latency mode
- [ ] Optional bounded continuous batching
- [ ] Prefix reuse and prompt-cache lifecycle

## Phase 5 — Server and Lumina integration

- [ ] HTTP server bound to `localhost:5150`
- [ ] OpenAI-compatible `/v1` inference surface
- [ ] Native management and telemetry API
- [ ] Streaming generation
- [ ] Tool-call grammar support
- [ ] Cancellation and session persistence
- [ ] Lumina Oracle dashboard controls
