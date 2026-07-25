# Architecture

## Layering

```text
Lumina / CLI / OpenAI-compatible API (planned)
                     |
               Runtime Engine
                 |       |
          Scheduler   Reference Execution
                     |       |
              Hybrid Cache  Sampler
                     |
              Backend Interface
                 |         |
        CPU reference   CUDA backend (planned)
                     |
       Core tensor / arena / scratch planning
                     |
      Qwen3.5 manifest / tokenizer / chat formatter
                     |
      Mapped GGUF model / GGML storage layouts
                     |
            GGUF structure reader
```

## Core modules

### `core`

Owns shared primitives: configuration, tensors, data types, shapes, strides, storage, errors, file mappings, and diagnostics. It must not depend on model-specific or backend-specific code.

The Phase 1A `Tensor` uses shared aligned storage. Copying or creating a view shares the underlying allocation; it does not copy payload data. Contiguous access is exposed through `std::span<float>`, while strided access uses checked multidimensional indexing.

Phase 1B adds `MemoryArena` for bounded monotonic allocation and `ScratchPlanner` for assigning reusable offsets to temporary buffers based on their liveness intervals. Planning remains separate from storage placement so the same plan can be consumed by CPU, CUDA, or hybrid execution.

Phase 1C adds `MappedFile`, a read-only RAII mapping used for immutable model weights. The mapping interface exposes bytes without introducing GGUF concepts into the core layer.

### `backend`

Defines execution capabilities. Backends own kernels, synchronization, and hardware feature detection. The CPU backend is the numerical correctness reference. CUDA support remains optional at compile time.

Phase 1A deliberately favors obvious scalar implementations over clever optimization. Later SIMD, quantized, and CUDA kernels must be testable against the same API and expected outputs.

### `model`

Owns model metadata, mapped weight views, and later graph construction. File-format parsing stays isolated from execution so GGUF compatibility does not become Oracle's internal architecture.

`GgufReader` parses GGUF v2/v3 structure, typed metadata, and tensor descriptors. `GgmlTypeLayout` describes how each supported storage type packs rows. `MappedGgufModel` validates payload ranges and exposes immutable `GgufTensorView` objects that point directly into the mapped file.

Phase 1C.1 keeps metadata export in the model layer. Full typed JSON is available for tokenizer and manifest construction, while compact summaries retain scalar values and replace large arrays with their element type and length. Terminal previews are bounded independently of the parsed metadata so inspection remains readable without discarding information.

Quantized decoding and architecture mapping will be layered above these views rather than embedded in the binary reader, metadata serializer, or mapping code.

Phase 1D adds `Qwen35Manifest`, a typed boundary between generic GGUF metadata and architecture execution. It resolves optional NextN/MTP layers separately from the shared backbone and validates tokenizer cardinality plus embedding dimensions when descriptors are available.

### `tokenizer`

Owns text segmentation, byte encoding, BPE merge ranks, token IDs, and decoding. `Qwen35Tokenizer` requires the GGUF-declared `gpt2` tokenizer model and `qwen35` pre-tokenizer, validates all 256 byte tokens, and keeps special-token parsing opt-in. Unicode categories are compiled data rather than locale-sensitive runtime calls.

### `runtime`

Coordinates engine lifecycle, backend selection, allocation, prompt ingestion, decode steps, cancellation, and telemetry.

The runtime layer also owns `format_qwen35_chat`, a typed implementation of the target model's message, reasoning, tool-call, and tool-response framing. It deliberately does not embed a general Jinja interpreter.

Phase 1E adds the stateful numerical reference layer. `KvCache` owns bounded full-attention keys and values. `SsmState` owns causal-convolution history plus the recurrent Gated DeltaNet matrix. `HybridCache` assigns the correct state type to every backbone block, while `plan_qwen35_cache` computes exact reference memory without allocation. Reference kernels cover embedding lookup, interleaved Qwen3.5 RoPE, causal grouped-query attention, causal depthwise convolution, and recurrent gated-delta updates. `Sampler` remains independent of model math.

`TinyHybridModel` composes these primitives into a deterministic synthetic decoder. It exists to prove state lifecycle and cached-versus-replayed parity; it is not the real Qwen3.5 architecture.

`ReferencePipeline` is the original stateless vertical slice through the runtime and backend layers:

```text
input
  -> RMSNorm
  -> up projection matmul
  -> SiLU
  -> down projection matmul
  -> residual add
  -> softmax
```

It is not intended to represent a full transformer block. Its purpose is to prove that tensor allocation, shape validation, backend dispatch, and composed execution work together.

### `scheduler`

Starts with a bounded single-request capacity. Future policies may support continuous batching, latency classes, and bounded concurrency without changing model math or backend interfaces.

## API surfaces

Oracle will eventually expose two distinct surfaces:

1. **OpenAI-compatible inference API** for Lumina and other clients.
2. **Native Oracle management API** for model lifecycle, memory planning, telemetry, scheduler state, and advanced controls.

The configured default endpoint is `127.0.0.1:5150`. The current engine records and reports that endpoint but does not yet bind a network socket.

## Near-term interfaces

- `IBackend`: hardware execution and tensor operations
- `Tensor`: shape, stride, dtype, and shared storage metadata
- `MemoryArena`: bounded aligned execution storage with telemetry
- `ScratchPlanner`: backend-neutral temporary-buffer layout
- `MappedFile`: immutable mapped file bytes with RAII ownership
- `GgufReader`: validated file structure, typed metadata, and tensor descriptors
- GGUF metadata serializers: full filtered exports and compact inspection summaries
- `GgmlTypeLayout`: block and byte layout for GGML storage types
- `MappedGgufModel`: validated model mapping and tensor registry
- `Qwen35Manifest`: validated architecture and MTP/backbone metadata
- `Qwen35Tokenizer`: Unicode-aware byte-level BPE encode/decode
- `format_qwen35_chat`: native message, reasoning, and tool prompt framing
- `KvCache`, `SsmState`, and `HybridCache`: bounded state ownership
- `plan_qwen35_cache`: exact reference-state memory planning
- reference execution kernels: embeddings, RoPE, attention, convolution, and gated delta
- `Sampler`: deterministic logit filtering and token selection
- `TinyHybridModel`: cached stateful end-to-end correctness fixture
- `GgufTensorView`: immutable zero-copy weight bytes
- `Model`: future architecture metadata and weight ownership policy
- `Engine`: lifecycle, configuration, and status reporting
- `Scheduler`: request admission and ordering
- `ReferencePipeline`: composed correctness smoke path

## Current non-goals

- Decoding F16, BF16, or quantized weight values
- Real quantized-checkpoint execution
- Production architecture tensor binding
- HTTP serving
- CUDA graph capture
- Paged or quantized KV cache
- Production-grade SIMD matmul

Those features should be added only after the reference interfaces remain stable under tests and benchmarks.
