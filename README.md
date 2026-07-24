# Oracle Inference Engine

Oracle is a local-first inference runtime aimed at efficient transformer execution on constrained consumer hardware. Phase 1B adds predictable temporary-memory planning and the first GGUF model-file foundation while retaining the correctness-first CPU path from Phase 1A.

## Current status — Phase 1B

Oracle is not yet a language-model runner or HTTP server. It now provides:

- C++20 core library and CLI
- 64-byte aligned F32 tensor storage
- Shape, stride, dtype, ownership, and bounds metadata
- Zero-copy contiguous, reshaped, and strided tensor views
- CPU reference kernels for elementwise operations, matmul, RMSNorm, SiLU, and softmax
- A synthetic transformer-like reference pipeline
- A monotonic aligned memory arena with marks, rewinds, scoped scratch lifetimes, and telemetry
- A deterministic scratch-buffer planner with liveness-based range reuse
- A dependency-free GGUF v2/v3 reader for:
  - file headers
  - scalar and nested-array metadata
  - alignment metadata
  - tensor names, dimensions, GGML type identifiers, and offsets
- `oracle-gguf-inspect` for text or JSON model inspection without loading weights
- Structured text and JSON engine status
- Default future server endpoint: `http://127.0.0.1:5150`
- Correctness tests and a dependency-free benchmark harness
- Optional CUDA build hook, still intentionally stubbed

## Build and test

```bash
cmake -S . -B build \
  -DORACLE_BUILD_TESTS=ON \
  -DORACLE_BUILD_EXAMPLES=ON \
  -DORACLE_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the CLI status output:

```bash
./build/oracle-cli
./build/oracle-cli --json
```

Run the vertical-slice demo and benchmark:

```bash
./build/oracle-reference-pipeline
./build/oracle-bench --rows 64 --inner 256 --columns 256 --iterations 10
```

Inspect a GGUF file without allocating its tensor payloads:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf
./build/oracle-gguf-inspect /path/to/model.gguf --json
```

The Phase 1B parser reads model structure only. Memory mapping, weight decoding, architecture mapping, tokenization, and generation are intentionally deferred.

## CUDA

CUDA remains opt-in until the CPU interfaces and expected numerical behavior are stable:

```bash
cmake -S . -B build-cuda -DORACLE_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

The current CUDA translation unit is only a build hook. Phase 1B does not claim GPU execution yet.

## Design principles

1. **Local-first:** optimize for single-user workstations before datacenter throughput.
2. **Predictable memory:** allocation and cache behavior must be inspectable and bounded.
3. **Backend independence:** runtime policy must not be tangled with CUDA implementation details.
4. **Measure before optimizing:** every major optimization needs a benchmark and fallback path.
5. **Incremental correctness:** begin with a simple CPU reference implementation, then accelerate.
6. **Compatibility plus native control:** the future OpenAI-compatible API stays separate from Oracle's richer management and telemetry API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/PHASE_1A.md`](docs/PHASE_1A.md), [`docs/PHASE_1B.md`](docs/PHASE_1B.md), and [`docs/ROADMAP.md`](docs/ROADMAP.md).
