# Phase 1E — Reference execution primitives

Phase 1E moves Oracle from prompt construction into stateful numerical execution. It does not yet run the quantized 4B checkpoint, but it establishes the CPU reference behavior and bounded state layouts that Phase 2 will connect to real GGUF weights.

## Reference kernels

`reference_kernels` adds:

- checked embedding lookup for batched token IDs
- Qwen3.5 partial RoPE with interleaved temporal, height, and width frequency selection
- single-token grouped-query causal attention over a bounded KV cache
- causal depthwise convolution state updates
- the recurrent Gated DeltaNet update used by Qwen3.5 linear-attention blocks

The Gated DeltaNet step follows the model's recurrent reference path:

1. L2-normalize query and key rows.
2. Apply the inverse-square-root key-width scale to the query.
3. Decay the recurrent key/value matrix.
4. Read the current value estimate from the matrix and key.
5. Apply the beta-scaled delta correction.
6. Write the key/delta outer product back into recurrent state.
7. Read the output using the query.

All calculations are scalar F32 CPU reference operations. They prioritize determinism and inspectability over speed.

## Hybrid cache and memory planning

Qwen3.5 interleaves three Gated DeltaNet blocks with one full-attention block. `HybridCache` therefore assigns each backbone block one of two bounded state types:

- `SsmState`: causal-convolution history plus recurrent Gated DeltaNet matrix
- `KvCache`: keys and values for the full-attention blocks

`qwen35_ssm_layout` derives the target state shape from the manifest:

```text
key_heads             = qwen35.ssm.group_count
value_heads           = qwen35.ssm.time_step_rank
key_head_dimension    = qwen35.ssm.state_size
value_head_dimension  = qwen35.ssm.inner_size / value_heads
convolution_channels  = 2 * key_width + value_width
```

For the validated 4B model this resolves to 16 key heads, 32 value heads, 128-wide key/value heads, and 8,192 causal-convolution channels.

`plan_qwen35_cache` computes exact F32 reference-state memory without allocating it. At 4,096 tokens, the 32-block target backbone requires eight KV caches and 24 SSM states, totaling 321,912,832 bytes in the reference layout.

Inspect a real model without allocating its cache:

```bash
./build/oracle-cache-plan /path/to/model.gguf --tokens 4096
./build/oracle-cache-plan /path/to/model.gguf --tokens 4096 --json
```

This is a correctness and planning baseline. Later cache formats may use lower precision, paging, tiering, or backend-specific packing.

## Sampling

`Sampler` provides:

- greedy decoding when temperature is zero
- temperature scaling
- top-k filtering
- top-p nucleus filtering
- deterministic seeded sampling
- candidate count and selected probability telemetry

The sampler rejects NaN-only or otherwise invalid logits instead of silently producing a token.

## Tiny hybrid reference model

`TinyHybridModel` is a deterministic, synthetic two-mixer decoder used to exercise the complete stateful path:

```text
token ID
  -> embedding lookup
  -> RMSNorm
  -> causal convolution
  -> recurrent gated-delta update
  -> gated SSM projection + residual
  -> RMSNorm
  -> Q/K/V projections
  -> partial RoPE
  -> grouped-query causal attention + KV cache
  -> gated output projection + residual
  -> RMSNorm
  -> tied-embedding logits
  -> sampler
```

Its dimensions and weights are deliberately tiny and generated from a fixed seed. It is not a compressed Qwen3.5 model and is not evidence that the real checkpoint can execute yet. Its purpose is to verify cache lifecycle, recurrent state, causal behavior, logits, sampling, and replay determinism before quantized weight loading is introduced.

Run the fixture:

```bash
./build/oracle-reference-decode
./build/oracle-reference-decode --json
./build/oracle-reference-decode \
  --tokens 1,5,2 --steps 4 \
  --temperature 0.8 --top-k 4 --top-p 0.9 --seed 5150 --json
```

The CLI replays the complete sequence and reports the maximum logit difference. The expected reference result is zero.

## Validation

Phase 1E tests cover:

- embedding rows and token bounds
- standard and interleaved multimodal RoPE coordinates
- RoPE norm preservation
- grouped-query attention and bounded KV-cache behavior
- causal depthwise convolution history
- hand-calculated recurrent gated-delta updates
- real-shape Qwen3.5 cache planning
- hybrid block-state typing and invalid access rejection
- greedy, top-k, top-p, and seeded sampling
- tiny-model cached decode versus full replay
- deterministic multi-token generation

The suite is validated with GCC, Clang, AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection.

## Upstream references

The state dimensions and numerical reference behavior were checked against:

- Hugging Face Transformers `modeling_qwen3_5.py`
- Hugging Face Transformers `configuration_qwen3_5.py`
- the current Unsloth Qwen3.5-4B configuration
- llama.cpp's current Qwen3.5 tokenizer and model integration

Oracle implements its own dependency-free reference code; no upstream implementation is copied into the runtime.

## Deliberate limits

Phase 1E does not:

- decode F16, BF16, Q5_K, or Q6_K model weights
- bind real Qwen3.5 tensor names to execution operators
- execute the full 32-block target checkpoint
- implement the architecture's production MLP, attention Q/K normalization, or output-gate loading from GGUF
- execute the optional MTP head
- provide SIMD, CUDA, HTTP serving, or production generation speed

Those tasks begin in Phase 2, using Phase 1E as the numerical and state-management oracle.
