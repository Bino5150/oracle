# Oracle Phase 2D — Complete Qwen3.5 Backbone and Logits

## Scope

Phase 2D composes the landed Phase 2C block executors into one complete scalar
CPU reference forward pass for a single real Qwen3.5/Qwopus token.

The phase:

1. decodes one real embedding row;
2. executes every bound backbone block in order;
3. preserves each of the 24 recurrent and 8 attention state owners;
4. applies `output_norm.weight`;
5. applies the tied token-embedding projection or an explicit output head;
6. returns exactly `vocabulary_size` finite logits;
7. captures every staged block output and the final normalized hidden state;
8. labels decoded-F32, Q8_K-reference, F32-cache, and pinned-F16-cache contracts;
9. supports strict diagnostic projection overrides for full-backbone contract
   decomposition;
10. stops before prompt ingestion, autoregressive generation, EOS handling, or
    MTP execution.

For the production Qwopus checkpoint the expected shape is:

```text
backbone blocks: 32
recurrent state owners: 24
attention state owners: 8
hidden width: 2,560
logits: 248,320
```

## Runtime API

The public Phase 2D entry point is:

```cpp
Qwen35ForwardResult execute_qwen35_reference_token(
    const Qwen35Manifest& manifest,
    const Qwen35Weights& weights,
    std::uint32_t token_id,
    HybridCache& state,
    RopePosition position,
    bool capture_block_outputs = true,
    const Qwen35ForwardOverrides* overrides = nullptr);
```

`Qwen35ForwardResult` owns the complete logits vector. Its trace preserves:

- token ID and position;
- embedding;
- every block output with block index and family;
- final RMSNorm output;
- tied-versus-explicit output-head status;
- explicit numerical-contract labels.

The function validates the complete manifest, binding, state-owner, position,
and override contract before state mutation where possible. It verifies that the
hybrid state advances exactly one token after the complete backbone.

## Numerical contracts

Oracle permanently preserves this default execution contract:

```text
weights: decoded Q5_K/Q6_K -> F32
activations: original F32
accumulation: scalar F32 reference
attention cache: semantic F32
```

The pinned TurboQuant production reference uses:

```text
weights/activations: Q5_K/Q6_K x Q8_K production dot kernels
attention cache: F16
```

These are separately labeled contracts. Phase 2D does not loosen tolerances or
change Oracle's scalar path to imitate the production runtime.

For derived-math validation, `Qwen35ForwardOverrides` accepts strict named
projection captures for individual blocks and an optional complete logits
capture. Every override is preflighted for block index, name, shape, duplicate
assignment, and finiteness before execution begins.

## CLI

```text
oracle-qwen35-forward <model.gguf> --token-id <id> [--position 0]
  [--logits-f32 <path>]
  [--trace-json <path>]
  [--override-f32 <block>:<trace-name>=<path>]...
  [--override-logits-f32 <path>]
  [--override-projection-contract <label>]
  [--override-cache-contract <label>]
  [--json | --full-json]
```

The Phase 2D CLI starts with fresh hybrid state and therefore accepts position
zero only. The library API remains stateful and requires the supplied text
position to equal the hybrid state's current sequence length.

A normal real-token run is:

```bash
./build-gcc/oracle-qwen35-forward \
  "$BASE_MODEL" \
  --token-id 9419 \
  --logits-f32 model-reports/qwopus-token9419-oracle-logits.txt \
  --trace-json model-reports/qwopus-token9419-oracle-forward.json
```

`--json` emits staged vectors plus a complete logits summary. `--full-json`
also embeds all logits values; the dedicated F32 file is preferred for complete
vector comparison.

## Independent comparison

`tools/compare_qwen35_forward_captures.py` compares:

- embedding;
- every staged block output;
- final RMSNorm;
- complete logits;
- argmax token;
- top-20 token-ID overlap.

It reports maximum absolute error, maximum relative error, relative L2 error,
and the first tolerance failure for every stage.

Two modes are explicit:

```text
same-contract
  Both captures must have identical execution projection and cache labels.

override-bridge
  Oracle must report diagnostic external overrides, and the override-source
  projection/cache labels must exactly equal the independent capture's
  execution labels.
```

A contract mismatch exits separately instead of being disguised as a numerical
mismatch.

## Deterministic tests

The synthetic Phase 2D fixture proves:

- recurrent and attention owners remain distinct;
- the complete backbone advances all state by one token;
- final RMSNorm is applied;
- tied output projection produces the expected logits;
- the complete logits count is preserved;
- MTP remains unexecuted;
- default F32/cache labels are stable;
- diagnostic override-source labels are stable;
- invalid override names and sizes are rejected before state mutation;
- position and block-family mismatches are rejected;
- summary and full JSON traces are valid and differentiated.

## Required real-model landing gates

Phase 2D is not landed until all of the following pass from
`~/oracle-release/`:

1. exact-base `git apply --check`;
2. GCC Release and complete CTest matrix;
3. Clang Release and complete CTest matrix;
4. ASan, UBSan, and leak detection;
5. clean-patch rebuild;
6. complete source-package rebuild;
7. real 32-block base-checkpoint forward for token 9419 at position zero;
8. exactly 248,320 finite Oracle logits;
9. staged independent capture comparison;
10. final RMSNorm comparison;
11. complete logits comparison under an explicitly matched contract;
12. separate Q8_K/F16 override-bridge comparison where used;
13. commit and push by Bino;
14. GitHub Actions green.

If the first staged block output differs, preserve both captures and stop at the
first divergence. Do not hide accumulated error by comparing only final logits.

## Deliberate limits

Phase 2D does not:

- tokenize or ingest a prompt;
- run more than the requested single-token forward contract in the CLI;
- perform an autoregressive decode loop;
- sample or decode text;
- handle EOS or stop sequences;
- execute the optional MTP/NextN head;
- add SIMD, threading, CUDA, or production-speed claims;
- change Oracle's semantic F32 attention cache;
- change Oracle's internal recurrent-state layout;
- mutate GitHub.
