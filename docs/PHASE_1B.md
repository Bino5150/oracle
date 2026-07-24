# Phase 1B — Predictable memory and GGUF foundations

Phase 1B adds the first pieces Oracle needs to load real model files without surrendering control of memory behavior.

## Memory arena

`oracle::core::MemoryArena` owns one aligned block and serves monotonic allocations from it. It provides:

- explicit capacity and base alignment
- per-allocation alignment
- marks and rewinds
- scoped scratch lifetimes through `ScopedArenaMark`
- current, peak, and allocation-count telemetry
- deterministic out-of-memory behavior through `std::bad_alloc`

The arena does not run constructors or destructors. It is intended for trivially managed tensor buffers, temporary kernel workspaces, and other execution-lifetime storage.

## Scratch planner

`oracle::core::ScratchPlanner` accepts named buffers with liveness intervals. The planner reuses byte ranges when lifetimes do not overlap and returns a deterministic `ScratchPlan` containing offsets and the required peak allocation.

This is deliberately backend-neutral. CPU and CUDA implementations can consume the same plan while deciding independently where the backing storage lives.

## GGUF metadata parser

`oracle::model::GgufReader` now reads GGUF v2 and v3 headers, metadata, arrays, and tensor descriptors. It does not load tensor payloads or decode quantized weights yet.

The parser includes safety limits, duplicate-name checks, alignment validation, boolean validation, nested-array depth limits, and file-boundary checks. The implementation follows the upstream GGUF specification and assumes little-endian model files.

Inspect a model without loading its weights:

```bash
./build/oracle-gguf-inspect /path/to/model.gguf
./build/oracle-gguf-inspect /path/to/model.gguf --json
```

## Deliberate boundaries

Phase 1B does not yet provide:

- memory mapping
- tensor payload size validation for every GGML type
- quantized decoding
- architecture-specific model loading
- tokenizer loading
- inference from GGUF weights

Those are staged for Phase 2 so the file parser remains independently testable.
