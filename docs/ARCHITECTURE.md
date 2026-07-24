# Architecture

## Layering

```text
Lumina / CLI / OpenAI-compatible API (planned)
                     |
               Runtime Engine
                 |       |
          Scheduler   Reference Pipeline
                     |
              Backend Interface
                 |         |
        CPU reference   CUDA backend (planned)
                     |
             Core tensor/storage
```

## Core modules

### `core`

Owns shared primitives: configuration, tensors, data types, shapes, strides, storage, errors, and diagnostics. It must not depend on model-specific or backend-specific code.

The Phase 1A `Tensor` uses shared aligned storage. Copying or creating a view shares the underlying allocation; it does not copy payload data. Contiguous access is exposed through `std::span<float>`, while strided access uses checked multidimensional indexing.

### `backend`

Defines execution capabilities. Backends own kernels, synchronization, and hardware feature detection. The CPU backend is the numerical correctness reference. CUDA support remains optional at compile time.

Phase 1A deliberately favors obvious scalar implementations over clever optimization. Later SIMD, quantized, and CUDA kernels must be testable against the same API and expected outputs.

### `model`

Owns model metadata and, later, graph construction and weight mapping. File-format parsing stays isolated from execution so GGUF compatibility does not become Oracle's internal architecture.

### `runtime`

Coordinates engine lifecycle, backend selection, allocation, prompt ingestion, decode steps, cancellation, and telemetry.

`ReferencePipeline` is the first vertical slice through the runtime and backend layers:

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

The configured default endpoint is `127.0.0.1:5150`. Phase 1A records and reports that endpoint but does not yet bind a network socket.

## Near-term interfaces

- `IBackend`: hardware execution and tensor operations
- `Tensor`: shape, stride, dtype, and shared storage metadata
- `Model`: immutable model metadata and weight ownership
- `Engine`: lifecycle, configuration, and status reporting
- `Scheduler`: request admission and ordering
- `ReferencePipeline`: composed correctness smoke path

## Current non-goals

- Parsing GGUF
- Tokenization and sampling
- Real transformer attention
- HTTP serving
- CUDA graph capture
- Paged or quantized KV cache
- Production-grade SIMD matmul

Those features should be added only after the reference interfaces remain stable under tests and benchmarks.
