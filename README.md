# Oracle Inference Engine

Oracle is a local-first inference runtime aimed at efficient hybrid-model execution on constrained consumer hardware. Phase 1E adds stateful CPU reference execution: embeddings, Qwen3.5 RoPE, grouped-query causal attention, Gated DeltaNet recurrent state, deterministic sampling, and exact cache-memory planning.

## Current status — Phase 1E

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
- full machine-readable metadata export with exact-key and prefix filters
- compact metadata summaries in normal inspection JSON so tokenizer arrays do not overwhelm reports
- bounded human-readable previews for large arrays and long strings
- validated `Qwen35Manifest` extraction, including 33-total/32-backbone MTP separation
- dependency-free Qwen3.5 GPT-2 byte-level BPE tokenization
- compiled Unicode 15.1 letter, mark, number, and whitespace classification
- explicit special-token parsing and optional special-token skipping on decode
- native Qwen3.5 system, reasoning, tool-call, and tool-response chat formatting
- `oracle-tokenize` manifest, encode, decode, and chat modes
- checked embedding lookup and Qwen3.5 partial/interleaved RoPE
- bounded full-attention KV caches and hybrid per-block state ownership
- causal depthwise-convolution history for linear-attention blocks
- recurrent Gated DeltaNet CPU reference updates
- grouped-query single-token causal attention
- exact Qwen3.5 F32 reference cache planning without allocation
- greedy, temperature, top-k, top-p, and seeded sampling
- deterministic tiny hybrid decoder with cached-versus-replayed parity
- `oracle-cache-plan` and `oracle-reference-decode` CLIs
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

Inspect GGUF structure without mapping tensor payloads:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf
./build/oracle-gguf-inspect /path/to/model.gguf --json
```

Export complete metadata values without tensor descriptors:

```bash
mkdir -p model-reports
./build/oracle-gguf-inspect /path/to/model.gguf --metadata --json \
  > model-reports/model-metadata.json
./build/oracle-gguf-inspect /path/to/model.gguf \
  --metadata-key general.architecture --json
./build/oracle-gguf-inspect /path/to/model.gguf \
  --metadata-prefix tokenizer.ggml. --json
```

Text metadata output uses bounded previews. JSON metadata mode preserves complete array and string values for tokenizer and model-manifest tooling.

Map and validate the real tensor payloads:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf --verify
./build/oracle-gguf-inspect /path/to/model.gguf --verify --json
./build/oracle-gguf-inspect /path/to/model.gguf --tensor token_embd.weight
```

Inspect the Qwen3.5 manifest and tokenize text:

```bash
./build/oracle-tokenize /path/to/model.gguf --manifest
./build/oracle-tokenize /path/to/model.gguf --encode "Hello Oracle" --json
./build/oracle-tokenize /path/to/model.gguf \
  --chat-system "You are Lumina." \
  --chat-user "Hello Oracle." \
  --tokenize-chat --json
```

Plan the reference cache for a real Qwen3.5 GGUF without allocating it:

```bash
./build/oracle-cache-plan /path/to/model.gguf --tokens 4096
./build/oracle-cache-plan /path/to/model.gguf --tokens 4096 --json
```

Run the deterministic tiny hybrid execution fixture:

```bash
./build/oracle-reference-decode
./build/oracle-reference-decode --temperature 0.8 --top-k 4 --top-p 0.9 --json
```

Phase 1E executes synthetic F32 reference weights and state, but it still does not decode or run the real quantized checkpoint.

## CUDA

CUDA remains opt-in until the CPU interfaces and expected numerical behavior are stable:

```bash
cmake -S . -B build-cuda -DORACLE_ENABLE_CUDA=ON
cmake --build build-cuda -j
```

The current CUDA translation unit is only a build hook. Phase 1E does not claim GPU execution yet.

## Design principles

1. **Local-first:** optimize for single-user workstations before datacenter throughput.
2. **Predictable memory:** allocation and cache behavior must be inspectable and bounded.
3. **Backend independence:** runtime policy must not be tangled with CUDA implementation details.
4. **Measure before optimizing:** every major optimization needs a benchmark and fallback path.
5. **Incremental correctness:** begin with a simple CPU reference implementation, then accelerate.
6. **Compatibility plus native control:** the future OpenAI-compatible API stays separate from Oracle's richer management and telemetry API.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/PHASE_1A.md`](docs/PHASE_1A.md), [`docs/PHASE_1B.md`](docs/PHASE_1B.md), [`docs/PHASE_1C.md`](docs/PHASE_1C.md), [`docs/PHASE_1C1.md`](docs/PHASE_1C1.md), [`docs/PHASE_1D.md`](docs/PHASE_1D.md), [`docs/PHASE_1E.md`](docs/PHASE_1E.md), and [`docs/ROADMAP.md`](docs/ROADMAP.md).
