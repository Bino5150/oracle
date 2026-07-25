# Phase 2B — Production Qwen3.5 tensor binding

Phase 2B turns the validated GGUF tensor registry into a typed, non-owning Qwen3.5/Qwopus weight contract. It does not copy mapped payloads and does not execute model math.

## Bound model structure

The binder validates and exposes:

- token embeddings;
- final normalization;
- tied or explicit output projection;
- all 32 backbone blocks;
- recurrent versus full-attention block classification;
- per-block input and post-attention normalization;
- dense MLP gate, up, and down projections;
- gated full-attention Q/K/V/output projections and Q/K norms;
- recurrent QKV/gate projections, convolution, time-step, decay, alpha/beta, normalization, and output tensors;
- an optional, separately owned NextN/MTP block.

The base Qwopus checkpoint is expected to bind 426 tensors. The known MTP checkpoint adds 15 tensors at `blk.32`, while preserving the same 32-block backbone.

## Validation

Every bound tensor is checked for:

- exact name;
- exact rank and dimensions derived from `Qwen35Manifest`;
- a supported storage class;
- exact byte geometry;
- unique assignment to one logical role.

Missing and malformed tensors produce role-specific errors. Unrecognized tensors are preserved in the deterministic binding report rather than silently discarded.

## Diagnostic CLI

```text
oracle-qwen35-bind <model.gguf> [--json]
```

The report includes block classification, required and optional tensor counts, tied-output status, mapped bytes, per-block totals, MTP discovery, and unexpected tensors.

## Deliberate limits

Phase 2B does not:

- execute a recurrent or attention block;
- dequantize complete matrices eagerly;
- implement optimized matvec or matmul;
- produce logits or generated text;
- execute the MTP head.

Those boundaries remain Phase 2C and later work.
