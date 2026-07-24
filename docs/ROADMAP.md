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

## Phase 1B — Memory planning and transformer primitives

- [ ] Monotonic arena allocator
- [ ] Reusable scratch-buffer planner
- [ ] Allocation and peak-memory telemetry
- [ ] RoPE reference kernel
- [ ] Causal masked attention reference kernel
- [ ] Embedding lookup
- [ ] Minimal sampler
- [ ] Tiny purpose-built model format

## Phase 2 — GGUF compatibility

- [ ] GGUF metadata parser
- [ ] Memory-mapped weights
- [ ] F16, BF16, and Q8 reference decoding
- [ ] Architecture mapping for one model family
- [ ] Quantization adapters separated from core tensor representation
- [ ] Tokenizer integration

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
