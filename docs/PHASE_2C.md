# Phase 2C — Single real Qwen3.5 block execution

Phase 2C connects the landed production tensor contract to Oracle's scalar CPU numerical path. It executes one production recurrent block or one production full-attention block for one token and emits named intermediate tensors for differential comparison.

## Implemented execution

### Recurrent/Gated DeltaNet block

The reference path performs:

1. input RMSNorm;
2. QKV, output-gate, beta, and alpha projections;
3. beta sigmoid;
4. biased alpha softplus and learned log-decay;
5. causal depthwise convolution and SiLU;
6. Q/K/V split and tiled Q/K head repetition matching GGUF's reordered V-head layout;
7. Q/K L2 normalization and scalar Gated DeltaNet update;
8. per-head RMSNorm multiplied by SiLU of the output gate;
9. recurrent output projection;
10. attention residual;
11. post-attention RMSNorm;
12. dense parallel SwiGLU MLP;
13. final residual.

### Full-attention block

The reference path performs:

1. input RMSNorm;
2. joint query/output-gate projection and per-head split;
3. Q/K/V projections;
4. per-head Q/K RMSNorm;
5. Qwen3.5 multimodal/interleaved RoPE;
6. single-token grouped-query causal attention;
7. sigmoid output gating;
8. attention output projection;
9. attention residual;
10. post-attention RMSNorm;
11. dense parallel SwiGLU MLP;
12. final residual.

## Mapped storage adapters

The CPU reference adapter now accepts F32, F16, BF16, Q5_K, and Q6_K mapped tensors through one interface. F32 is decoded with explicit little-endian reads, preserving the existing alignment-safe mapped-byte policy.

No production matrix is eagerly dequantized. Matrix-vector products decode one storage row at a time. Small vectors and the depthwise convolution kernel are decoded into temporary F32 arrays.

## Diagnostic CLI

```text
oracle-qwen35-block <model.gguf> <block-index>
  (--token-id <id> | --input-f32 <path>)
  [--position <n>]
  [--output-f32 <path>]
  [--json]
```

Examples:

```bash
./build-gcc/oracle-qwen35-block \
  "$BASE_MODEL" 0 \
  --token-id 9419 \
  --position 0 \
  --json > model-reports/qwopus-block0-oracle.json
```

For an attention block, use the exact block input captured from the independent runtime:

```bash
./build-gcc/oracle-qwen35-block \
  "$BASE_MODEL" 3 \
  --input-f32 model-reports/qwopus-block3-input.txt \
  --position 0 \
  --json > model-reports/qwopus-block3-oracle.json
```

The standalone Phase 2C CLI currently accepts only position 0 because it constructs fresh zero recurrent/KV state. Nonzero positions require a future state-import path or a caller that supplies the preceding state.

`--token-id` is a convenience for block 0. Feeding an embedding directly into a later block is an isolated synthetic probe, not a complete backbone prefix.

## Trace comparison

The JSON trace preserves full F32 values and uses callback-style names aligned with the pinned upstream Qwen3.5 graph where practical.

Compare an Oracle trace with an independently generated trace:

```bash
python3 tools/compare_qwen35_block_captures.py \
  model-reports/qwopus-block0-oracle.json \
  model-reports/qwopus-block0-reference.json \
  --atol 1e-5 --rtol 1e-5
```

The independent reference must not call Oracle's decoder or executor. A pinned llama.cpp/beellama tensor callback is the most direct comparison. A Hugging Face reference is also acceptable only after its grouped V-family head order is transformed into the tiled order stored by GGUF.

## Success criteria before landing

- complete GCC and Clang CTest matrices;
- ASan, UBSan, and leak detection;
- clean patch and package rebuilds;
- real block 0 recurrent execution on Qwopus;
- real block 3 attention execution using the exact independent block input;
- matching intermediate tensors within recorded tolerances;
- no claim of full-backbone logits or generation.

## Deliberate limits

Phase 2C does not:

- execute all 32 backbone blocks;
- perform prompt prefill;
- produce final logits;
- generate text;
- execute MTP;
- add SIMD, threading, CUDA, or production-speed kernels;
- define a persistent state ABI.
