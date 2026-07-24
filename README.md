# Oracle Inference Engine

Oracle is a local-first inference runtime aimed at efficient transformer execution on constrained consumer hardware. Phase 1A establishes a correctness-first CPU execution path that optimized and CUDA kernels can later use as their reference.

## Current status — Phase 1A

Oracle is not yet a language-model runner or HTTP server. It now provides a tested miniature execution stack:

- C++20 core library and CLI
- 64-byte aligned F32 tensor storage
- Shape, stride, dtype, ownership, and bounds metadata
- Zero-copy contiguous, reshaped, and strided tensor views
- CPU reference kernels for:
  - elementwise add
  - elementwise multiply
  - rank-2 matrix multiplication
  - RMSNorm
  - SiLU
  - softmax over the final dimension
- A synthetic transformer-like reference pipeline
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

Run the vertical-slice demo:

```bash
./build/oracle-reference-pipeline
```

Run a small F32 matrix-multiplication benchmark:

```bash
./build/oracle-bench --rows 64 --inner 256 --columns 256 --iterations 10
```

The benchmark emits one JSON object so future runs can be captured and compared without scraping decorative output.

## CUDA

CUDA remains opt-in until the CPU interfaces and expected numerical behavior are stable:

```bash
cmake -S . -B build-cuda -DORACLE_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

The current CUDA translation unit is only a build hook. Phase 1A does not claim GPU execution yet.

## Design principles

1. **Local-first:** optimize for single-user workstations before datacenter throughput.
2. **Predictable memory:** allocation and cache behavior must be inspectable and bounded.
3. **Backend independence:** runtime policy must not be tangled with CUDA implementation details.
4. **Measure before optimizing:** every major optimization needs a benchmark and fallback path.
5. **Incremental correctness:** begin with a simple CPU reference implementation, then accelerate.
6. **Compatibility plus native control:** the future OpenAI-compatible API stays separate from Oracle's richer management and telemetry API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/PHASE_1A.md`](docs/PHASE_1A.md), and [`docs/ROADMAP.md`](docs/ROADMAP.md).
