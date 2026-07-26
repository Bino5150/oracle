# Phase 2C contract bridge

Oracle's scalar reference matvec deliberately evaluates decoded Q5_K/Q6_K rows against the original F32 activation. The pinned TurboQuant CPU runtime evaluates the same stored rows against a Q8_K-quantized activation. Both paths are correct, but they do not produce tensor-by-tensor equality after the first projection.

The Phase 2C contract bridge decomposes validation without changing Oracle's scalar numerical contract.

## Matvec override API

`Qwen35MatvecOverrides` supplies named projection outputs to a block executor. Every supplied name is validated against the block family, duplicate names are rejected, all values must be finite, and the replacement count must equal the mapped matrix row count.

The diagnostic CLI accepts repeatable overrides:

```text
--override-f32 <trace-name>=<path>
```

The files contain whitespace-delimited F32 values. The trace still records each replacement under its normal tensor name.

### Recurrent names

- `linear_attn_qkv_mixed`
- `z`
- `beta`
- `alpha`
- `linear_attn_out`
- `ffn_gate`
- `ffn_up`
- `ffn_out`

### Attention names

- `Qcur_full`
- `Kcur_projected`
- `Vcur_projected`
- `attn_output`
- `ffn_gate`
- `ffn_up`
- `ffn_out`

## Validation decomposition

1. **Decoded-F32 matrix contract** — GGML official dequantized-F32 matvec versus Oracle. Already passed exactly for real Q5_K/Q6_K production tensors.
2. **Production matrix contract** — GGML Q5_K/Q6_K × Q8_K versus independently captured TurboQuant projections.
3. **Post-projection contract** — inject the captured projection outputs into Oracle and compare every derived convolution, normalization, state, attention, gating, residual, and SwiGLU tensor.

Injected tensors are compared too, proving the bridge consumed the intended values. They are not counted as independent operator evidence; the derived tensor section is the actual post-projection gate.

## Scope

The bridge is diagnostic only. It does not replace Oracle's reference matvec, add a production Q8_K kernel, define a public serialized state ABI, or alter generation behavior.
