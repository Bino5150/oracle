# Phase 1A — CPU Reference Runtime

Phase 1A turns the initial framework scaffold into a small, executable numerical runtime.

## Delivered

### Tensor foundation

- F32 dtype metadata
- Checked non-zero shapes
- Row-major contiguous strides
- 64-byte aligned allocation by default
- Shared storage ownership
- Zero-copy slices and reshapes
- Explicit strided views
- Bounds-checked multidimensional access
- Rejection of raw contiguous spans for non-contiguous views

### CPU reference kernels

- `add`
- `multiply`
- `matmul` for rank-2 tensors
- `rms_norm` over the final dimension
- `silu`
- `softmax` over the final dimension

The implementations are intentionally straightforward. Their first responsibility is deterministic correctness, not maximum throughput.

### Vertical slice

The reference pipeline composes the kernel set into a transformer-like path and verifies that each output row is a valid probability distribution.

### Diagnostics

The engine reports both text and JSON status, including:

- readiness
- Oracle version
- selected backend
- context length
- concurrency capacity
- configured server host and port

Oracle's default port is `5150`.

### Benchmarking

`oracle-bench` measures F32 matrix multiplication and emits structured JSON containing dimensions, iteration count, average latency, approximate GFLOP/s, and a checksum.

## Exit criteria

Phase 1A is complete when:

- GCC and Clang builds succeed with warnings enabled
- all correctness tests pass
- the reference pipeline runs end to end
- the benchmark produces stable structured output
- no CUDA dependency is required for the CPU path

## Next implementation slice

Phase 1B should add:

1. monotonic arena allocation for temporary tensors
2. reusable scratch-buffer planning
3. tensor storage telemetry
4. RoPE reference kernel
5. causal attention reference kernel
6. benchmark result capture and comparison
