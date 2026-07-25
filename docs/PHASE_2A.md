# Phase 2A — Storage decoding and quantized reference math

Phase 2A connects Oracle's validated GGML storage layouts to scalar numerical values without allowing file-format representation to leak into the core tensor type.

## Storage adapters

`StorageRowView` is a non-owning, validated row adapter containing:

- GGML storage type;
- logical element count;
- immutable row bytes.

Construction verifies the exact row size through the existing GGML layout registry. The first supported production storage types are F16, BF16, Q5_K, and Q6_K.

`GgufTensorView` rows can now be exposed through the same adapter without copying mapped payloads. The adapter derives the logical row width from the first GGML dimension, validates total row geometry, bounds-checks the requested row, and returns the exact immutable byte subspan for that row.

## Reference decoding

The decoder provides:

- bit-exact F16-to-F32 conversion, including subnormals, signed zero, infinities, and NaNs;
- bit-exact BF16-to-F32 conversion;
- alignment-safe Q5_K super-block decoding;
- alignment-safe Q6_K super-block decoding;
- complete row decoding for one or more 256-value quantization blocks.

Mapped bytes are read explicitly in little-endian order. The implementation does not cast mapped storage to packed C++ structures, avoiding alignment and strict-aliasing assumptions.

## Quantized math foundation

The scalar CPU reference backend adds:

- storage-row dot products;
- contiguous row-major matrix-vector multiplication;
- strict dimension and byte-size validation;
- block-local temporary decoding rather than whole-matrix expansion.

This is the correctness baseline for later direct integer dot products, SIMD dispatch, prefill matrix multiplication, and CUDA kernels.

## Mapped-row inspection

`oracle-storage-row` maps a GGUF model, selects a named tensor row, decodes it through the Phase 2A reference path, and prints either a compact preview or complete JSON values. This gives the validation workflow a direct bridge from real Qwopus payload bytes to independently comparable row fixtures.

```text
oracle-storage-row <model.gguf> <tensor-name> <row-index> [--json]
```

## Known-answer fixtures

The Phase 2A suite includes independently constructed byte fixtures for:

- F16 and BF16 edge values;
- Q5_K affine scales, minima, low nibbles, and high bits;
- Q6_K signed scales, low nibbles, and upper two bits;
- reference dot products;
- two-row quantized matvec;
- truncated rows, unsupported formats, and output-size mismatch rejection.

## Deliberate limits

Phase 2A does not yet:

- bind production Qwen3.5 tensor names to operators;
- execute a production model block;
- provide SIMD or CUDA kernels;
- provide optimized Q5_K/Q6_K integer dot products;
- claim full Qwopus logits or text generation.

Those boundaries remain Phase 2B and later work.
