# Oracle Inference Engine

Oracle is a local-first inference runtime aimed at efficient transformer execution on constrained consumer hardware. Phase 1C adds safe, zero-copy access to GGUF tensor payloads while retaining the correctness-first CPU path and predictable-memory foundation from Phases 1A and 1B.

## Current status — Phase 1C

Oracle is not yet a language-model runner or HTTP server. It now provides:

- C++20 core library and CLI
- 64-byte aligned F32 tensor storage
- shape, stride, dtype, ownership, and bounds metadata
- zero-copy contiguous, reshaped, and strided tensor views
- CPU reference kernels for elementwise operations, matmul, RMSNorm, SiLU, and softmax
- a synthetic transformer-like reference pipeline
- a monotonic aligned memory arena with marks, rewinds, scoped scratch lifetimes, and telemetry
- a deterministic scratch-buffer planner with liveness-based range reuse
- a dependency-free GGUF v2/v3 reader for metadata and tensor descriptors
- read-only mapped-file ownership with RAII cleanup and move semantics
- a GGML storage-layout registry with checked row and tensor byte-size calculation
- `MappedGgufModel` validation for tensor bounds, block shapes, offsets, and overlaps
- zero-copy named tensor views into mapped GGUF payloads
- `oracle-gguf-inspect` metadata, verification, tensor lookup, and JSON modes
- structured text and JSON engine status
- default future server endpoint: `http://127.0.0.1:5150`
- correctness tests and a dependency-free benchmark harness
- optional CUDA build hook, still intentionally stubbed

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

Inspect GGUF metadata without mapping tensor payloads:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf
./build/oracle-gguf-inspect /path/to/model.gguf --json
```

Map and validate the real tensor payloads:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf --verify
./build/oracle-gguf-inspect /path/to/model.gguf --verify --json
./build/oracle-gguf-inspect /path/to/model.gguf --tensor token_embd.weight
```

Phase 1C validates and exposes weight bytes but does not decode quantized values or execute a real model yet.

## CUDA

CUDA remains opt-in until the CPU interfaces and expected numerical behavior are stable:

```bash
cmake -S . -B build-cuda -DORACLE_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

The current CUDA translation unit is only a build hook. Phase 1C does not claim GPU execution yet.

## Design principles

1. **Local-first:** optimize for single-user workstations before datacenter throughput.
2. **Predictable memory:** allocation and cache behavior must be inspectable and bounded.
3. **Backend independence:** runtime policy must not be tangled with CUDA implementation details.
4. **Measure before optimizing:** every major optimization needs a benchmark and fallback path.
5. **Incremental correctness:** begin with a simple CPU reference implementation, then accelerate.
6. **Compatibility plus native control:** the future OpenAI-compatible API stays separate from Oracle's richer management and telemetry API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/PHASE_1A.md`](docs/PHASE_1A.md), [`docs/PHASE_1B.md`](docs/PHASE_1B.md), [`docs/PHASE_1C.md`](docs/PHASE_1C.md), and [`docs/ROADMAP.md`](docs/ROADMAP.md).
