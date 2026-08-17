# Oracle Phase 2F — Qwen3.5 MTP/NextN Contract, Binding, and Speculative Verification

**Status: LANDING CANDIDATE — SLICE 7 (CLOSURE).** Phase 2F builds from landed commit
`162f0451bd4fd9710d0cc401fd4a6e4ed044c7df` (`1.1.0-phase2e`). Phase 2E's
validated guarded Qwen3.5 autoregressive generation is treated as immutable
baseline behavior and is unmodified by any Phase 2F slice. Phase 2F is not
landed. Slice 1 was real-checkpoint forensics, reference capture, and
contract verification. Slice 1B added reviewed Q8_0 scalar decode support
and closed the one binding blocker Slice 1 found. Slice 2 executed exactly
one real Qwen3.5 NextN/MTP prediction through Oracle's decoded-F32 scalar
reference path. Slice 3 added one-token speculative verification (depth
fixed at 1, no chaining). Slice 4 added multi-draft MTP chaining and
sequential partial-acceptance verification (a bounded chain of D0..D(depth-1)
proposals, verified one at a time against freshly-forwarded target state,
first-rejection semantics, no rollback needed or used). Slice 5 added a
second, independent verification strategy: batched (multi-token) target
evaluation with canonical-prefix rollback via an exact `HybridCache`
boundary truncate/restore, cross-validated bit-for-bit against Slice 4's
unmodified sequential verifier. Slice 6 integrated this machinery into
`Qwen35GenerationSession`, guarded and off by default. MTP never changes
canonical generation: it only *predicts* upcoming tokens ahead of time (via
a disposable shadow copy of canonical state), and every committed token —
MTP on or off — still passes through the exact same real forward and real
greedy sampler call Phase 2E's existing per-token commit path always used,
with an internal-consistency assertion that the real sample agrees with
MTP's prediction. **Slice 7 (this update) is Phase 2F's closure slice**: it
closes the one literal coverage gap Slice 6 left open (a real, verified
`[A, EOS, C]` decision batch with `A != EOS`), runs a broader real MTP
ON/OFF canonical-parity matrix across diverse prompts and every supported
depth, finalizes this document, and prepares the version-1.2.0-phase2f
landing candidate. No shadow-state promotion, no bonus-token consumption,
no adaptive depth, no stochastic speculative sampling, and no performance
claim anywhere in Phase 2F. Everything above the Slice 7 section is
prior slices' original record,
updated only where a later slice
supersedes it.

## Slice 1 scope and central finding

Slice 1 set out to add "the minimum model-level API necessary for Oracle to
represent an MTP-capable Qwen3.5 checkpoint" and real MTP tensor binding.
Forensic investigation of the codebase found that **this contract and
binding logic already exist**, landed in Phase 2B
(`include/oracle/model/qwen35_manifest.hpp`, `qwen35_weights.hpp`,
`src/model/qwen35_manifest.cpp`, `qwen35_weights.cpp`): `Qwen35Manifest`
already carries `nextn_predict_layers`/`has_mtp()`, and `Qwen35Weights`
already carries an optional `Qwen35MtpWeights mtp` populated by
`bind_qwen35_weights()`, with full validation (missing/duplicate/wrong
rank/dimension/type/layer-count all rejected deterministically). Slice 1
therefore did not re-implement this contract; it verified it against a real
MTP checkpoint and a real reference implementation, found it structurally
and semantically correct, and closed the loop with new forensics-informed
tests. **No changes were made to `include/oracle/model/qwen35_manifest.hpp`,
`qwen35_manifest.cpp`, `qwen35_weights.hpp`, or `qwen35_weights.cpp`.**

The one substantive new fact this slice establishes: the real
`unsloth/Qwen3.5-2B-MTP-GGUF` checkpoint (Q5_K_M) stores its single required
`blk.24.nextn.eh_proj.weight` tensor as **Q8_0**, a storage type Oracle's
`storage_decode.cpp` did not decode at the time. Oracle's existing binder
rejected this deterministically and by name
(`blk.24.nextn.eh_proj.weight: unsupported storage type q8_0 for matrix
weight`) via `allowed_storage_type()` in `qwen35_weights.cpp`. Per the
task's explicit instruction not to "smuggle a new quantization
implementation into this slice," Q8_0 decoding was deliberately **not**
added in Slice 1, and was carried forward as the one open item.

**Slice 1B closed this blocker.** Q8_0 scalar storage decoding was added to
Oracle's generic storage layer (not an MTP-specific hack), validated
bit-identical against the pinned `beellama`/GGML reference on synthetic
data and on real `eh_proj` rows, and the real 2B MTP checkpoint now binds
in full — 335/335 tensors, 0 unexpected. See "Slice 1B — Q8_0 storage
support and MTP binding closure" below for the complete record. The
narrative in this section and in "Binding validation" immediately below is
preserved as Slice 1's original finding for provenance.

## Primary 2B MTP checkpoint identity

- Path: `/home/bino/lumina llm models/Qwen3.5-2B-Q5_K_M-MTP.gguf`
- Size: 1,471,868,320 bytes
- SHA-256: `b8d558161010664c469b59efeed318e8a64267b1aecdaabeb21a1c44e21aac22`
- GGUF version: 3
- Quantization: Q5_K_M mix (Q5_K/Q6_K/F32 backbone, one Q8_0 exception — see
  above), `general.quantized_by = Unsloth`, `general.file_type = 17`
- Architecture string: `general.architecture = qwen35`
- `quantize.imatrix.*`: imatrix-calibrated quantization
  (`Qwen3.5-2B-GGUF/imatrix_unsloth.gguf`, 186 entries, 77 chunks)

### Geometry (from `qwen35.*` GGUF metadata, authoritative)

| key | value |
|---|---|
| `qwen35.block_count` | 25 |
| `qwen35.nextn_predict_layers` | 1 |
| backbone block count (derived) | 24 |
| `qwen35.context_length` | 262144 |
| `qwen35.embedding_length` | 2048 |
| `qwen35.feed_forward_length` | 6144 |
| `qwen35.attention.head_count` | 8 |
| `qwen35.attention.head_count_kv` | 2 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.attention.layer_norm_rms_epsilon` | 9.99999997e-07 |
| `qwen35.rope.dimension_count` | 64 |
| `qwen35.rope.dimension_sections` | [11, 11, 10, 0] |
| `qwen35.rope.freq_base` | 10000000 |
| `qwen35.ssm.conv_kernel` | 4 |
| `qwen35.ssm.state_size` | 128 |
| `qwen35.ssm.group_count` | 16 |
| `qwen35.ssm.time_step_rank` | 16 |
| `qwen35.ssm.inner_size` | 2048 |
| `qwen35.full_attention_interval` | 4 |
| vocabulary size | 248320 (gpt2 tokenizer, `qwen35` pre-tokenizer) |
| `tokenizer.ggml.eos_token_id` | 248046 |
| `tokenizer.ggml.padding_token_id` | 248055 |
| `tokenizer.chat_template` | present, 7991 bytes, Qwen3.5-style Jinja with `<think>` reasoning support |

Tensor alignment: 32 (GGUF default), 335 tensors total, `data_offset =
10963104`. Every tensor's `absolute_offset` is a multiple of the declared
alignment (verified). Whole-file type distribution: f32 × 176, q5_K × 113,
q6_K × 45, **q8_0 × 1** (the sole occurrence is `blk.24.nextn.eh_proj.weight`).

Full raw dumps are preserved in
`model-reports/phase2f-slice1-mtp-baseline-20260816/` (not committed):
`2b-mtp-gguf-descriptor.json` (metadata + tensor list),
`2b-mtp-gguf-verified.json` (mmap-verified, with byte offsets/sizes).

## External reference implementation identity

- Repository: `/home/bino/beellama` (fork of llama.cpp, `Anbeeld/beellama.cpp`)
- Commit: `a620cbd481d16d3ccb5c8c96fa2fbd70191bea38`, branch `v0.3.2`, clean tree
- Binary: `build/bin/llama-cli`, `build/bin/llama-server` — `version: 10274
  (a620cbd48)`, built with GNU 14.2.0 for Linux x86_64
- Backend: CUDA-enabled Release build (`GGML_CUDA=ON`, `GGML_CPU=ON`,
  `CMAKE_BUILD_TYPE=Release`)
- MTP support confirmed via `--spec-type
  none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,...` CLI surface and
  via source: `src/models/qwen35.cpp` contains a complete `graph_mtp` class;
  by contrast, the other local checkout (`/home/bino/llama-cpp-turboquant`,
  `TheTom/llama-cpp-turboquant`, same base commit `5aeb2fdbe26cd4c534c6fa15de73cb5749bd0403`)
  has zero MTP-related code in its `qwen35.cpp` — it was **not** used as the
  reference. Neither local llama.cpp checkout was pulled, reset, or modified
  to obtain this behavior; `beellama` already had it.

### Model vs. reference-implementation distinction

Source-grounded reading of `beellama`'s Qwen3.5 MTP pipeline (commit
`a620cbd48`) separates architecture-level facts a from-scratch
implementation must reproduce from this engine's own staging/engineering
choices:

**(A) Architecture/checkpoint-level facts:**
- GGUF contract: `block_count` = trunk + NextN layers;
  `qwen35.nextn_predict_layers` = trailing NextN block count; NextN tensors
  live at `blk.<i>.nextn.*` for the trailing block index(es).
- Required per-NextN-block tensors: `eh_proj [2n_embd, n_embd]`, `enorm
  [n_embd]`, `hnorm [n_embd]`, plus a full independent decoder block's
  attention+FFN weights (own norms, fused-gate Q, GQA K/V, QK-norm, output
  projection, SwiGLU FFN) — structurally identical to a normal Qwen3.5
  full-attention layer, numerically distinct.
- Optional/tie-able tensors: `embed_tokens`, `shared_head_head`,
  `shared_head_norm` — default to the main model's `tok_embd`/`output`/
  `output_norm` when the checkpoint omits them. The real 2B checkpoint uses
  exactly this: `embed_tokens` and `shared_head_head` are absent (tied);
  `shared_head_norm` is present.
- Forward math: `h_norm = RMSNorm(hnorm, h_target_after_output_norm)`;
  `e_norm = RMSNorm(enorm, embed(candidate_token))`; `concat(e_norm,
  h_norm)` **in that order** → `eh_proj` → standard pre-norm attention block
  (own weights) → SwiGLU FFN → final norm → LM head. The hidden state fed
  into the MTP head is the trunk's **last hidden state after its own final
  `output_norm`**, not a pre-norm residual value.
- MTP requires, at every drafting/verification step, both a token id and a
  hidden-state row simultaneously (dual input); the hidden state used is
  always shifted one position (h at position p pairs with the token to be
  predicted at p+1).
- Acceptance is generic greedy resample-and-compare against the draft
  (standard speculative decoding); MTP only supplies the proposal
  distribution and a confidence gate for how far to draft.
- Qwen3.5's hybrid linear/full-attention layout is specifically
  architected for cheap partial (RS) rollback of rejected draft state,
  rather than full KV checkpoint/restore.

**(B) beellama/llama.cpp-specific staging choices (not architectural
requirements):** the entire `llama_set_embeddings_nextn` /
`llama_get_embeddings_nextn*` / `llama_get_ctx_other` API surface lives in
`src/llama-ext.h`, explicitly marked "a staging header ... everything here
should be considered WIP" — it is not part of the stable `include/llama.h`;
"MTP-ness" is modeled as a property of an entire second `llama_context`
(`ctx_type = LLAMA_CONTEXT_TYPE_MTP`), not a per-call graph parameter; the
masked/unmasked NextN-embedding-extraction split and the
`need_full_h_nextn` graph-optimization-suppression flag are this engine's
specific mechanism for satisfying "need h at every position"; the draft
head's own autoregressive sampling uses a fixed top-10 sampler independent
of the user's real sampler chain; and the hard `n_layer_nextn == 1` limit in
`graph_mtp`'s constructor is an engineering limitation of this fork, not a
property of the Qwen3.5 architecture. Oracle's contract (below) reproduces
only the (A) facts.

## Oracle's optional MTP contract (Phase 2B, unmodified by this slice)

`Qwen35Manifest.nextn_predict_layers` (default 0) and `has_mtp()` (`return
nextn_predict_layers != 0`) make MTP presence a checkpoint fact, not a
compile-time or runtime flag. `load_qwen35_manifest()` reads
`qwen35.nextn_predict_layers` as an **optional** key defaulting to 0, so
ordinary non-MTP checkpoints are unaffected; when present, it additionally
requires at least one `blk.<backbone_block_count>.nextn.*` tensor to exist
(rejecting metadata/tensor disagreement).

`Qwen35Weights.mtp` is `std::optional<Qwen35MtpWeights>`, populated by
`bind_qwen35_weights()` only when `manifest.has_mtp()`. The binder currently
requires `nextn_predict_layers == 1` (rejects other values by name).
`Qwen35MtpWeights` carries its own attention/MLP weights plus
`embedding_hidden_projection` (`eh_proj`), `embedding_norm` (`enorm`),
`hidden_norm` (`hnorm`) as required tensors, and `embed_tokens`,
`shared_head`, `shared_head_norm` as optional (nullable) tensors. The MTP
block is never included in `Qwen35Weights.blocks` (the backbone block
list) — it is tracked exclusively through `weights.mtp`.

### Binding validation (unmodified, verified against real data)

`bind_qwen35_weights()` rejects, by exact tensor name and reason: a missing
required tensor (backbone or MTP); a tensor whose dimensions don't match
the manifest-derived expected shape; a storage type not in `{F32, F16,
BF16}` for vectors, or not in `{F32, F16, BF16, Q5_K, Q6_K}` for matrices; a
duplicate tensor name; a mapped byte size inconsistent with declared
geometry; a `total_block_count` that doesn't equal `backbone_block_count +
nextn_predict_layers`; and (MTP-specific) a `nextn_predict_layers` value
other than 1. All of this was exercised against the real 2B MTP checkpoint:
manifest parsing succeeds (`has_mtp=true`, `nextn_predict_layers=1`,
`backbone_block_count=24`); the 24 backbone blocks (18 recurrent + 6
attention, `full_attention_interval=4`) and the MTP block's own
attention/MLP/norm tensors (11 of the 15 `blk.24.*` tensors, all
non-`nextn.*`) bind without error; binding then fails deterministically at
`blk.24.nextn.eh_proj.weight` with `unsupported storage type q8_0 for
matrix weight` — the Q8_0 blocker described above. `enorm`/`hnorm`/
`shared_head_norm` were independently confirmed shape/type-correct via raw
forensic inspection (all F32, matching expected shapes) but were not
reached by the binder in this run, since `bind_mtp()` is fail-fast and
throws at the first invalid required tensor.

*(Slice 1B note: this paragraph describes Slice 1's original binding
attempt, preserved for provenance. As of Slice 1B, `eh_proj`'s Q8_0 storage
type is supported and this fixture binds successfully in full — see "Slice
1B" below.)*

`tests/test_phase2f_slice1.cpp` (new in Slice 1, extended in Slice 1B)
closes the loop: it adds binder-level tests for a missing required MTP
tensor, a malformed MTP tensor dimension, and a `nextn_predict_layers != 1`
rejection. It also adds a synthetic-topology test
(`test_real_qwen35_2b_mtp_topology`) that reconstructs the real checkpoint's
block-kind pattern (24 backbone blocks, interval 4 → 18 recurrent/6
attention) at small synthetic dimensions and asserts `required_tensor_count
== 334`, `optional_tensor_count == 1`, `bound_tensor_count == 335` —
**exactly** the real file's 335-tensor count, with exactly one optional
tensor present (`shared_head_norm`), confirming Oracle's contract predicts
the real checkpoint's tensor topology exactly from manifest geometry alone.
(Slice 1 originally also added a Q8_0-on-`eh_proj` *rejection* test here;
Slice 1B updated it in place to assert *acceptance* instead, and added a
replacement test proving a genuinely unsupported type is still rejected —
see "Slice 1B" below for the full test inventory.)

## Same-checkpoint MTP OFF vs. MTP ON reference capture

Both runs used the identical real 2B MTP checkpoint, identical raw
ChatML-formatted prompt (`<|im_start|>system\nYou are a helpful
assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of
France?<|im_end|>\n<|im_start|>assistant\n`), `beellama`'s `llama-server`
via its native `/completion` endpoint, `temperature=0`, `seed=42`,
`-ngl 99`, `-c 4096`, single sequence, no vision input, no other
speculative method. Full residency confirmed both runs (no CPU-offload
warnings; MTP-ON log: "estimated memory usage of MTP context is 56.02 MiB";
GPU used ≈1981 MiB of 3714 MiB during the MTP-ON run — no contamination of
the kind that affected the earlier 4B result).

| | MTP OFF (`--spec-type none`) | MTP ON (`--spec-type draft-mtp --spec-draft-n-max 3`) |
|---|---|---|
| generated tokens | `[248068, 271, 248069, 271, 760, 6511, 314, 9338, 369, 2972, 57590, 159034, 248046]` | **identical** |
| generated text | `<think>\n\n</think>\n\nThe capital of France is **Paris**.` | **identical** |
| tokens_predicted / tokens_evaluated | 13 / 26 | 13 / 26 |
| finish condition | natural EOS (`stop_type: eos`) | natural EOS (`stop_type: eos`) |
| prompt / predicted t/s | 73.24 / 53.80 | 58.70 / 62.29 |
| draft stats | n/a | `draft_n=9, draft_n_accepted=9` (100% acceptance) |

Under deterministic greedy sampling, MTP OFF and MTP ON produce the
**bit-identical accepted target token sequence** — the correctness
condition the task required verifying before any further speculative-decoding
work is meaningful. All 9 drafted tokens were accepted for this
low-perplexity prompt; no rejection/rollback path was exercised by this
capture. Timing is observational only (single short generation, not a
performance benchmark) and is not used to make any Oracle performance claim.

## Real NextN intermediate capture

A small disposable harness (`model-reports/.../nextn_capture.cpp`, not part
of Oracle or beellama source) linked against `beellama`'s `libllama.so`
directly, using the staging `llama_set_embeddings_nextn` /
`llama_get_embeddings_nextn_ith` API (declared in the private
`src/llama-ext.h`, not the stable public header) to decode the same fixed
prompt on a plain target context with `embeddings_nextn` enabled
(unmasked), and read out the target's `h_nextn` row at the last prompt
position (index 25, token id 198, i.e. immediately before generation
starts).

- Embedding length: 2048 (matches `qwen35.embedding_length`)
- Capture position: 25 (last prompt token, of 26 total — cross-validated:
  independent `llama_tokenize` in the harness also produced 26 tokens,
  matching the server's `tokens_evaluated=26`)
- Deterministic fingerprint (FNV-1a 64): `0xb99610ab1cb34658`
  (bit-identical across two independent runs)
- First 8 values: `2.31288671, 0.280516416, -12.0319777, 1.24401402,
  3.82044411, -1.44941938, 9.29557228, -3.61193562`
- Sum: `158.212357`, sum of squares: `25142.4642`

This captures the target-model hidden state associated with MTP (the exact
input `h` that `eh_proj` consumes after its own `hnorm`), for one fixed
position, as evidence only — it is **not** Oracle MTP execution. Capturing
the MTP head's own draft logits/token additionally (running the draft graph
itself) was attempted and explicitly deferred: doing so faithfully requires
replicating `common_speculative_impl_draft_mtp`'s internal `pending_h`
carryover state (the one-position-shifted hidden-state handoff across
`process()`/`draft()` calls), which is not exposed through the public/staging
C API and is non-trivial to reproduce correctly outside `common/speculative.cpp`
itself in a "small, disposable" harness. The clean avenue for Slice 2 is
linking against `beellama`'s `libllama-common.so` (already built, exports
`common_speculative_*`) rather than hand-rolling the batch/state management.

## Numerical contract separation

Oracle's permanent scalar numerical reference is unchanged by this slice:

```text
weights:         Q5_K/Q6_K decoded to F32
activations:     original F32
accumulation:    scalar F32 in deterministic index order
attention cache: semantic F32
```

MTP/NextN is treated as a new, separately-labelled numerical contract (not
yet defined numerically — no MTP execution exists in Oracle). The canonical
Phase 2D standalone fingerprint, `0x54825e50fa9398cf` (for
`Qwopus3.5-v3-4B.Q5_K_M.gguf`, sha256
`a66ea93e1fe0470cd3c405cd858955b59c9126d43088d77dd894e3f309363f04`, token ID
`9419`, position `0`), was re-verified bit-identical against a fresh GCC
Release build of this slice's tree.

## Slice 1B — Q8_0 storage support and MTP binding closure

### Why Q8_0 support was added

Slice 1 found exactly one blocker between Oracle and a fully-bound real
Qwen3.5-2B-MTP-GGUF: `blk.24.nextn.eh_proj.weight` is stored as Q8_0, a type
Oracle's `storage_decode.cpp` did not decode. Slice 1B's mandate was narrow
and explicit: add *reviewed, permanent* Q8_0 support to Oracle's *generic*
storage layer — not an MTP-specific special case — then retry the real
binding. Q8_0 now sits alongside F16/BF16/Q5_K/Q6_K as a storage encoding
Oracle's decode-to-F32 layer understands; the MTP binder/executor does not
know or care which of these backs a given matrix.

### Authoritative Q8_0 format contract

Source: pinned `beellama` (fork of llama.cpp), commit
`a620cbd481d16d3ccb5c8c96fa2fbd70191bea38`.

- Block struct (`ggml/src/ggml-common.h:325-329`):
  ```c
  #define QK8_0 32
  typedef struct {
      ggml_half d;       // delta (scale), 2-byte IEEE-754 half, little-endian
      int8_t  qs[QK8_0]; // 32 signed 8-bit quants, one byte each
  } block_q8_0;
  // sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0 == 2 + 32 == 34 bytes
  ```
- Dequantization (`ggml/src/ggml-quants.c:876-890`,
  `dequantize_row_q8_0`):
  ```c
  const float d = GGML_FP16_TO_FP32(x[i].d);
  for (int j = 0; j < 32; ++j) {
      y[i*32 + j] = x[i].qs[j] * d;
  }
  ```
  A pure per-block scale multiply — **no zero-point, no minimum**, unlike
  Q5_K/Q6_K's `d, dmin` pair. `GGML_FP16_TO_FP32` is the standard bit-exact
  IEEE-754 binary16→binary32 conversion (`ggml/src/ggml-impl.h:384-399`),
  algorithmically identical to the `f16_to_f32` Oracle already implements
  and had already validated via the Phase 2D/2E gates for Q5_K/Q6_K deltas.
- Block element count: 32 (`QK8_0`). Block byte size: 34. Quant
  signedness: signed 8-bit (`int8_t`, range −128..127), stored as one raw
  byte per value — no bit-packing, so no endianness ambiguity for the
  quants themselves. Scale endianness: little-endian 2-byte half, same
  convention GGUF already uses throughout (matches Oracle's existing
  `load_u16_le`). No alignment assumptions beyond the tensor-level 32-byte
  GGUF alignment Oracle already enforces.
- Oracle's `ggml_type.cpp` registry already carried the exactly-correct
  entry — `{8, "q8_0", 32, 34, true}` (type id, block elements, bytes per
  block, quantized) — predating this slice; it required no correction.

### Source changes (storage layer only)

- `include/oracle/model/storage_decode.hpp`: added `ggml_type_q8_0 = 8`,
  `q8_0_block_elements = 32`, `q8_0_block_bytes = 34` constants.
- `src/model/storage_decode.cpp`: added `decode_q8_0_block()` (explicit
  little-endian byte reads via the existing `load_u16_le`/`signed_byte_value`
  helpers — no `reinterpret_cast` to a packed `block_q8_0*`, consistent with
  Oracle's existing Q5_K/Q6_K decoders); added Q8_0 to the allowed-type list
  in `make_storage_row_view()`; added a Q8_0 dispatch branch in
  `decode_storage_row()` (block size 32, distinct from the K-quants' shared
  256-element `qk_k`).
- `src/backend/cpu/quantized_reference.cpp`: added a Q8_0 branch to
  `reference_storage_dot()` (32-element block loop, decode-then-accumulate,
  same scalar-F32 contract as the K-quant path). `reference_storage_matvec`,
  `reference_mapped_tensor_matvec`, and `make_storage_row_view(tensor, row)`
  needed **no changes** — they were already generic over the type registry
  and the storage decoder's allowed-type list, so Q8_0 support flowed
  through automatically once the two files above accepted it. No CUDA, no
  SIMD, no activation quantization, no Q8_K path — weight-storage decode
  only, per the task's hard exclusions.
- `src/model/qwen35_weights.cpp`: `allowed_storage_type()` now accepts
  `Q8_0` for `TensorRole::matrix` (alongside `Q5_K`/`Q6_K`). Vector policy
  (`{F32, F16, BF16}`) is **unchanged** — no known Qwen3.5 vector weight
  requires Q8_0, and this was verified rather than assumed (see synthetic
  tests below).

### Synthetic Q8_0 fixtures (hand-computed, independent of the decoder)

Added to `tests/test_phase2a.cpp` (the existing storage-decode test file),
following its established Q5_K/Q6_K fixture style exactly:

- `q8_0_fixture()`: a single hand-built 34-byte block, `d = 1.0` (chosen so
  `decoded[i] == qs[i]` exactly, making the expected output trivially
  hand-verifiable), covering zero, +1/−1, the int8 extremes (+127/−128), and
  a symmetric ±2..±14 ramp plus a trailing +15.
- `test_q8_0_known_answer()`: asserts the registry geometry (32
  elements/block, 34 bytes/block, `quantized=true`) and decodes the fixture
  against 32 independently-computed expected values.
- `test_q8_0_multi_block_row()`: two blocks with distinct deltas (1.0 and
  0.5) and distinct quant patterns, decoded as one row.
- `test_reference_dot_and_matvec()` (extended): Q8_0 scalar dot of the
  fixture against a ones-vector — expected value `14.0`, hand-derived from
  `sum(quants) * d` (the ±k pairs cancel; only the unpaired `0`, the
  `+127/−128` pair (`−1`), and the trailing `+15` survive: `0 + (−1) + 15 =
  14`).
- `test_q8_0_mapped_tensor_row_adapter()`: a two-row Q8_0 matrix through the
  same mmap-view adapter path used by Q5_K, verifying row count, exact
  byte-offset advancement (no copies), correct per-row dot values, and
  rejection of an out-of-range row and of inconsistent tensor byte
  geometry.
- `test_validation()` (extended): truncated-storage rejection (33 of 34
  bytes) and malformed row-width rejection (17 elements, not divisible by
  the 32-element block size).

None of these expected values were generated by calling Oracle's own
decoder — they were computed by hand from the format contract above.

### Independent differential validation vs. pinned beellama/GGML

A disposable harness (`model-reports/.../q8_0_differential.cpp`, not part
of Oracle or beellama source) linked Oracle's `liboracle_engine.a` against
beellama's `libggml-base.so` (which exports the real `dequantize_row_q8_0`)
and fed byte-identical raw rows into both. Every case was **bit-identical**
(all 32 or 4096 output floats equal by raw bit pattern, `max_abs_error =
0`, `l2_relative_error = 0`, no first-differing index):

| case | elements | bitwise-equal | result |
|---|---|---|---|
| synthetic single block | 32 | 32/32 | BIT_IDENTICAL |
| synthetic multi-block row (3 blocks) | 96 | 96/96 | BIT_IDENTICAL |
| real `blk.24.nextn.eh_proj.weight` row 0 | 4096 | 4096/4096 | BIT_IDENTICAL |
| real `blk.24.nextn.eh_proj.weight` row 1024 (middle) | 4096 | 4096/4096 | BIT_IDENTICAL |
| real `blk.24.nextn.eh_proj.weight` row 2047 (last) | 4096 | 4096/4096 | BIT_IDENTICAL |

This is the expected outcome for two implementations sharing the same
half-to-float conversion and the same multiplication ordering, and confirms
Oracle's Q8_0 decode is not merely close but numerically identical to the
authoritative GGML reference on real production data.

### Real `eh_proj` row known-answer record

- Checkpoint: `Qwen3.5-2B-Q5_K_M-MTP.gguf`, SHA-256
  `b8d558161010664c469b59efeed318e8a64267b1aecdaabeb21a1c44e21aac22`
- Tensor: `blk.24.nextn.eh_proj.weight`, shape `[4096, 2048]`, storage Q8_0
- Row 0 — raw bytes FNV1a64 `0x7d08957bada6dced`; decoded FNV1a64
  `0xa63be9c101978ef7`; first 4 values `-0.0201356411, 0.000951290131,
  0.00301241875, -0.0104641914`
- Row 1024 (middle) — raw bytes FNV1a64 `0x5dd271c62c4d5939`; decoded
  FNV1a64 `0xfd7ad57bb5c26c33`; first 4 values `0.00195801258,
  -0.00454258919, 0.00383770466, 0.00172305107`
- Row 2047 (last) — raw bytes FNV1a64 `0x69009b54792decd7`; decoded FNV1a64
  `0x97ba646f0a0d75e8`; first 4 values `0.00741553307, -0.00400048494,
  0.00234174728, -0.00331747532`
- All three rows: bit-identical to the independent GGML reference decode
  (see table above).

### Real MTP binding closure

`oracle-qwen35-bind` against the real 2B checkpoint now succeeds in full:

```text
architecture: qwen35, backbone_block_count: 24, recurrent: 18, attention: 6
nextn_predict_layers: 1, mtp_block_index: 24
required_tensor_count: 334, optional_tensor_count: 1, bound_tensor_count: 335
unexpected_tensor_count: 0, bound_bytes: 1460905216, tied_output: true
```

`bound_tensor_count = 335` matches the file's total tensor count exactly
(no unexpected tensors); `optional_tensor_count = 1` confirms only
`shared_head_norm` is present among the optional MTP tensors, with
`embed_tokens`/`shared_head_head` correctly absent and tied to the main
model, exactly as Slice 1's forensics predicted. `bound_bytes =
1460905216` matches the mmap-verified `validated_payload_bytes` recorded in
Slice 1's GGUF forensics exactly. Reproduced identically in a clean
detached-worktree reconstruction (see Build gates below).

`tests/test_phase2f_slice1.cpp` was extended with: `eh_proj` Q8_0 now
**accepted** (replacing the Slice 1 rejection test, whose original intent —
that the binder still discriminates supported from unsupported types — is
preserved by a new test using a still-genuinely-unsupported type, Q4_K);
Q8_0 still rejected for a *vector* MTP tensor (`enorm`), proving the policy
change is matrix-only and intentional, not a blanket relaxation; a
malformed/truncated Q8_0 `eh_proj` still rejected on byte-geometry grounds;
and a full real-scale (24-block) topology test with `eh_proj` as Q8_0
asserting the exact real-checkpoint counts (334/1/335/0 unexpected) end to
end.

### Continued absence of MTP execution

Slice 1B changes what Oracle can **bind** and **decode to F32**. It adds no
forward pass, no NextN logits, no draft generation, no speculative
acceptance/rejection, no cache branching, no scheduling, and no CUDA/SIMD
work. `Qwen35MtpWeights` after `bind_qwen35_weights()` is still just mapped
tensor views — nothing reads `eh_proj`/`enorm`/`hnorm` through the model's
math yet. The separation is exact: Slice 1B proves *the bytes decode
correctly*; a future slice must still prove *the MTP forward math is
correct*, which is an entirely separate, larger claim this slice makes no
attempt at. **Slice 2 (below) is that future slice.**

## Slice 2 — real Qwen3.5 NextN/MTP scalar execution

### Pinned `graph_mtp` execution contract (source-verified, not assumed)

Read end-to-end from `beellama` commit `a620cbd481d16d3ccb5c8c96fa2fbd70191bea38`,
`src/models/qwen35.cpp:660-817` (`graph_mtp`'s constructor). Exact operation
sequence, with residual/normalization points nailed down precisely (no
ambiguity left to "conceptual sketch"):

1. `tok_embd = embedding_lookup(candidate_token_id)` via `nextn.embed_tokens`
   if present, else tied to the backbone's `token_embd` (line 694). The real
   checkpoint has no `nextn.embed_tokens`, so this is always the tied path.
2. `h` is supplied externally as a second graph input (`inp->h`,
   `ggml_set_name(..., "mtp_h_input")`, line 702-704) — it is **not**
   recomputed inside the MTP graph. It is the **target backbone's own
   `h_nextn`**: the trunk's last hidden state taken *after* the trunk's own
   final `output_norm` (backbone `graph::graph`, line 287-290:
   `cur = build_norm(inpL, model.output_norm, ...); res->t_h_nextn = cur;`).
   A stale code comment elsewhere (`llama-context.cpp:6307`, "hidden state
   before the final output norm") is inaccurate relative to what the graph
   actually computes; the graph code is authoritative.
3. `h_norm = RMSNorm(hnorm, h)` (line 715) and, independently,
   `e_norm = RMSNorm(enorm, tok_embd)` (line 718) — plain RMSNorm on each
   input directly, no other preprocessing.
4. `concat = ggml_concat(e_norm, h_norm, dim=0)` (line 721) — **`e_norm`
   first**, `h_norm` second, along the feature dimension, giving the
   `2*n_embd`-wide input `eh_proj` expects.
5. `cur = eh_proj(concat)` (line 724) → back down to `n_embd`. **`inpSA =
   cur`** (line 727): the residual base for the attention sub-block is the
   *post-`eh_proj`* value, not the pre-projection `concat`/`h`/`tok_embd`.
6. From here the sequence is **line-for-line identical** to the backbone's
   `build_layer_attn`/`finish_block` (trunk `graph::build_layer_attn`,
   lines 350-430, and the shared residual/FFN tail every backbone
   full-attention block uses): pre-norm (`attn_norm`, the MTP block's own
   weight) → fused Q+gate projection (`wq`, split into `Qcur`/`gate` by the
   same interleaved-stride convention as the trunk) → per-head `Qcur`/`Kcur`
   RMSNorm → MRoPE on Q/K (same `rope_sections`/`freq_base` as the trunk —
   there is only one RoPE config in the manifest, shared by construction) →
   scaled-dot-product attention over the MTP block's *own* KV state → `cur =
   attn_out * sigmoid(gate)` → output projection (`wo`) → **residual add
   back to `inpSA`** (line 779, i.e. the post-`eh_proj` value) → post-attn
   norm → SwiGLU FFN (`up`/`gate`/`down`, `LLM_FFN_SILU`/`LLM_FFN_PAR`, same
   as the trunk's dense FFN) → **residual add back to the post-attention
   value** (line 794).
7. `head_norm_w = nextn.shared_head_norm` if present, else the backbone's
   `output_norm` (line 797-799) → `cur = RMSNorm(head_norm_w, cur)`. This
   result is exposed as `res->t_h_nextn` (line 804) for hypothetical
   chaining to a *second* MTP call — a **different tensor** from the
   backbone's own `h_nextn` that fed step 2, despite the identical name.
8. `cur = get_rows(cur, inp_out_ids)` selects the requested output
   position(s) (trivial/identity for Slice 2's single-token case).
9. `head_w = nextn.shared_head_head` if present, else the backbone's
   `output` tensor (line 809-811) — which is itself already tied to
   `token_embd` when the checkpoint has no explicit `output.weight`. The
   real checkpoint has neither `shared_head_head` nor an explicit
   `output.weight`, so this always resolves to the ordinary tied token
   embedding, confirming section 12's question directly from source.
10. `logits = head_w @ cur` → full vocabulary-wide draft logits.

**Architecture facts (A) vs. this engine's staging choices (B)** — the full
breakdown (llama_set_embeddings_nextn/llama_get_ctx_other staging API,
`LLM_CONTEXT_TYPE_MTP` as a whole-context switch, the masked/unmasked
extraction split, the dual token+embd `llama_batch` plumbing) is unchanged
from Slice 1's record above; nothing new in that split was needed for
execution, since Oracle receives `h`/`candidate_token_id`/position directly
as explicit function arguments rather than through any batch/context
abstraction.

### Reference fixture and intermediate captures

Same fixed prompt as Slice 1 (`<|im_start|>system\nYou are a helpful
assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of
France?<|im_end|>\n<|im_start|>assistant\n`, 26 tokens), seed position 25
(last prompt token, id `198`) — the exact position/token Slice 1 already
established feeds the first MTP draft step.

A disposable harness
(`model-reports/.../slice2-mtp-forward-capture/mtp_forward_capture.cpp`,
not part of Oracle or beellama source) used the standard ggml eval-callback
mechanism (`ggml_backend_sched_eval_callback`, the same mechanism
`examples/eval-callback`/`common/debug.cpp` use) to capture **every** named
intermediate `graph_mtp` produces, by running the real `ctx_dft` (MTP)
context with `llama_decode()` on a one-token batch carrying both
`token[0]=198` and `embd[0..2048)=`target `h_nextn`. One naming subtlety
required source reading to resolve: `llama_context::graph_get_cb()`
(`src/llama-context.cpp:7830-7853`) formats every `cb()` name as `"%s-%d"`
via `ggml_format_name` whenever the call passes `il >= 0` — which is true
for every MTP-block-scoped tensor (`il = hparams.n_layer() = 24`) except
the three graph-level outputs (`h_nextn`, `mtp_shared_head_norm`,
`result_output`, all called with `il = -1`). All 20 named tensors were
captured successfully once this was accounted for. Full raw F32 dumps and
fingerprints are preserved in that evidence directory (not committed).

One capture bug found and fixed during this work: the harness initially
compared Oracle's execution against `graph_mtp`'s own `"h_nextn"` output
tensor (identical to `mtp_shared_head_norm`, exposed only for hypothetical
chaining) instead of the **target backbone's** `h_nextn` (the actual input
to step 2 above, retrieved via `llama_get_embeddings_nextn_ith(ctx_tgt,
25)`). The harness now dumps the target's `h_nextn` separately as
`target_h_nextn.f32.raw`; using the wrong tensor as input made every
downstream comparison meaningless until corrected.

### Oracle MTP execution API

New, additive files: `include/oracle/runtime/qwen35_mtp.hpp`,
`src/runtime/qwen35_mtp.cpp`. `execute_qwen35_reference_mtp(manifest,
weights, Qwen35MtpInput{candidate_token_id, target_hidden_state, position},
capture_intermediates)` returns `Qwen35MtpResult{logits, argmax_token_id,
contracts, trace}`. It requires `manifest.has_mtp()` and bound
`weights.mtp`; it is a pure function of its explicit inputs — it does not
read or write any external state, is not wired into
`Qwen35GenerationSession`, and implements exactly one prediction (no draft
loop, no acceptance, no rollback).

Almost none of the MTP block's math is newly written. Source forensics
established (see above) that the block's own attention+FFN
sub-computation, residualed against `eh_proj`'s output, is architecturally
identical to an ordinary Qwen3.5 full-attention backbone block — so
`execute_qwen35_reference_mtp` constructs a `Qwen35BackboneBlockWeights`
view over `weights.mtp`'s attention/MLP/norm tensors and calls the already-
validated `execute_qwen35_attention_block_reference()` (Phase 2A/2B/2D/2E)
unchanged. New code is limited to: token embedding lookup + `enorm`/`hnorm`
RMSNorm + concatenation + `eh_proj`'s matvec (via the existing generic
`reference_mapped_tensor_matvec`, Q8_0-capable since Slice 1B) before the
block, and `shared_head_norm` + the tied output matvec after it. The small
RMSNorm/embedding-row helpers in the new file mirror
`qwen35_forward.cpp`'s private `rms_norm`/`embedding_row` formulas exactly
(verified byte-for-byte against that source) rather than modifying that
file — consistent with this project's per-file-private-helper convention
and its preference for minimal, additive diffs over touching validated
files.

### State ownership

`execute_qwen35_reference_mtp` does not accept any cache/state parameter at
all. The MTP block's single-position attention KV state (a `KvCache`
sized for exactly one entry) is created fresh inside the call and destroyed
when it returns. This is a stronger form of isolation than "a separate MTP
state object": there is no object for canonical code to accidentally share,
alias, or leak into `HybridCache`. `tests/test_phase2f_slice2.cpp`'s
`test_state_isolation_from_canonical_cache` constructs a real
`HybridCache`, executes an MTP prediction, and asserts the cache's
`sequence_length()` is unchanged.

### `eh_proj` and MTP block comparison against the real capture

Using the **real target `h_nextn`** as `target_hidden_state` (not Oracle's
own backbone forward — that is Phase 2D/2E's already-validated concern, out
of scope here; Slice 2 isolates the MTP-specific math from backbone
recomputation, per the taskblock's "Oracle receives exactly the same input
contract"), stage-by-stage comparison against the real capture
(`model-reports/.../slice2-mtp-forward-capture/oracle-vs-real-comparison.txt`):

| stage | result |
|---|---|
| `mtp_tok_embd`, `mtp_hnorm`, `mtp_enorm`, `mtp_concat` | **bit-identical** (max_abs=0, 100% bitwise-equal) |
| `mtp_eh_proj` onward (every block stage, `shared_head_norm`, full logits) | small, proportional divergence (relative L2 ≈0.3%–2.6%) |
| draft argmax token | **248068 — matches Oracle exactly, and matches the target's own greedy choice** (consistent with Slice 1's 100% draft-acceptance capture) |
| top-20 logit overlap | 19/20 |
| repeated execution | byte-identical across runs (deterministic) |

The divergence begins at **exactly** the first quantized-weight matvec
(`eh_proj`, Q8_0) and is consistent with (not larger than) accumulated
quantization/precision differences through every subsequent Q5_K/Q6_K-
weighted stage in the MTP block. This was verified by elimination, not
assumed: Slice 1B already proved Oracle's Q8_0 *decode* is bit-identical to
GGML's `dequantize_row_q8_0` on this exact tensor (including real
`eh_proj` rows), and `mtp_concat` (the matvec's other operand) is
bit-identical between Oracle and the real capture. With both operands
proven identical, the only remaining variable is the **matvec accumulation
method itself** — and beellama's real CUDA/production kernels for quantized
weight types are well known to requantize the F32 activation on the fly
(e.g. to Q8_1) for fast integer dot products, rather than decoding the
weight to F32 and accumulating in F32 as Oracle's reference deliberately
does. This is the **same class of numerical-contract difference** Phase
2C/2D already established and documented for Q5_K/Q6_K
(`docs/PHASE_2E.md`: "TurboQuant Q5_K/Q6_K x Q8_K projection math ... remain
separately labelled independent-production contracts") — not a new
phenomenon, not a bug, and explicitly not something to chase into bit-
identity by altering Oracle's scalar semantics (forbidden by this
taskblock's section 7) or by reimplementing beellama's CUDA kernels
(forbidden by section 21). `Qwen35MtpContractLabels` labels this
explicitly: `execution_projection = "decoded-f32-scalar"`,
`execution_attention_cache = "f32-semantic"` — the same permanent
contract labels the ordinary backbone already uses, extended to MTP.

### Shared output head

Confirmed directly from execution (not just from checkpoint/source
inspection): the real checkpoint's projection resolves to
`weights.output`, which is itself tied to `weights.token_embedding` (no
explicit `output.weight` tensor in this checkpoint). `logits.size() ==
248320` on every real and synthetic run. `test_shared_head_tied_output_behavior`
independently recomputes the final projection from the captured
`mtp_shared_head_norm` output using `weights.output` directly and asserts
it matches Oracle's own logits exactly, proving the tied fallback path
specifically (not merely that *some* projection executed).

### Synthetic tests

`tests/test_phase2f_slice2.cpp` (new), 11 hand-controlled tests, none
requiring the real checkpoint: MTP-absent rejection; incomplete MTP weights
rejected (exercises `execute_qwen35_reference_mtp`'s own validation, not
just the binder's); input-width mismatch; invalid position; non-finite
input rejection; a known-small `eh_proj` calculation with independently
hand-computed expected output (`RMSNorm`'s scale-invariance made an earlier
draft of this test vacuous — fixed by using distinct *weights*, not just
distinct input magnitudes, to make `enorm`/`hnorm` distinguishable);
`mtp_concat` ordering (`e_norm` first, `h_norm` second, with distinguishable
values proving the halves are not swapped); tied shared-head-output
behavior; state isolation from a real `HybridCache`; exact logits count;
and bit-identical determinism across repeated calls.

## Slice 3 — single-draft target verification + accept/reject state semantics

### Pinned depth-1 verification timeline (source-verified)

Read from `common/speculative.cpp` (`common_speculative_impl_draft_mtp::accept`,
lines 1043-1056) and `tools/server/server-context.cpp` (the `use_mtp_spec_accept`
block, lines 6929-7068), beellama commit `a620cbd48`.

**The central finding**: verification does **not** require a second target
forward. `common_sampler_sample_and_accept_n` (the real accept routine) walks
logits the target **already produced** from a batch it decoded *before* the
accept/reject decision — the real implementation's specific strategy is to
opportunistically fold the draft token into the same target batch as the
anchor (to get a "bonus" token for free when accepted), but the comparison
itself only ever needs the anchor position's own logits. Oracle's design
takes the simpler, architecturally-equivalent path this taskblock's own
"expected depth-1 semantic model" describes: compute the draft from `h_nextn`
separately, compare it against the target's already-produced logits at the
same position, decide, and only *then* forward the canonical token through
the target — never advancing target state speculatively in the first place.

This directly resolves the "does depth-1 need rollback" question. The real
implementation's `common_context_seq_rm(...)` calls after acceptance
(`server-context.cpp:7038-7041`) are **trimming speculative KV entries that
implementation already wrote optimistically** — a reference-implementation
*performance strategy*, not a model requirement. Since Oracle never writes
speculative entries before the decision, there is nothing to trim. **No
rollback machinery was added**, and none was needed: this is model semantics
(target authority, exactly one canonical token committed) versus reference
implementation strategy (speculative-advance-then-maybe-trim) versus Oracle's
design choice (decide-first-then-advance-once) — three distinct things, kept
distinct.

`common_speculative_impl_draft_mtp::accept()` itself only ever updates
`pending_h` (a hidden-state handoff for *chaining* a subsequent draft round)
— it never touches canonical target state either. It is out of scope here
since Slice 3 does not chain (draft depth fixed at 1).

Timeline, source-proven end to end:
1. Canonical target state exists (`HybridCache` after ordinary prompt/generation
   processing) before any proposal.
2. The target's own logits at the current position are already the next-token
   distribution — no extra step needed to obtain them.
3. `h_nextn` is available immediately after that same target forward (Slice 1/2:
   it is the trunk's last hidden state, post-`output_norm`).
4. MTP produces its draft token from `h_nextn` (Slice 2's
   `execute_qwen35_reference_mtp`), independent of and prior to any verification.
5. Verification is comparison only: `target_token = argmax(target logits)`,
   `accepted = (draft_token == target_token)`.
6. The canonical token (draft if accepted, target's own token if rejected) is
   forwarded through the ordinary target path exactly once, advancing
   `HybridCache` by exactly one position.
7. MTP's temporary state (Slice 2's ephemeral single-position `KvCache`,
   scoped to `execute_qwen35_reference_mtp`) is already gone by the time
   verification runs — accept or reject changes nothing about it, since it
   was never retained in the first place.

### Oracle verification API

New, additive files: `include/oracle/runtime/qwen35_mtp_verify.hpp` /
`src/runtime/qwen35_mtp_verify.cpp`, and
`include/oracle/runtime/qwen35_state_fingerprint.hpp` /
`src/runtime/qwen35_state_fingerprint.cpp`.

`Qwen35MtpProposal{draft_token, draft_logits, draft_position}` +
`verify_qwen35_mtp_proposal(proposal, target_logits, target_position) ->
Qwen35MtpDecision{draft_token, target_token, canonical_token, accepted,
target_position, draft_position, accounting}`. `draft_token` must be
self-consistent with `argmax(draft_logits)` (catches a caller passing a stale
token id) — this is what makes "invalid draft token" a real, distinct
validation case rather than a structurally-unreachable one. `verify_qwen35_mtp_proposal`
is a **pure function**: no cache/state parameter exists in its signature at
all, so canonical `HybridCache` is provably unreachable from it — a stronger
guarantee than "isolated state," there is simply no state to isolate.
Forwarding the resulting `canonical_token` through the target (advancing
`HybridCache`) is the caller's separate, subsequent step, done with the
already-existing `execute_qwen35_reference_token` — no new state-advancing
code was written for Slice 3 at all.

`Qwen35StateFingerprint` (diagnostic/test-only, explicitly not a serialized
persistent-state ABI, per the taskblock's own instruction) hashes every
attention block's K/V entries and every recurrent block's convolution
history and recurrent state, in block order, into one deterministic FNV1a64
`combined` value plus a per-block breakdown — this is what proves Lane A/B
canonical-state equivalence below without dumping full state into ordinary
logs.

`Qwen35MtpVerificationTrace{decision, target_state_before_fingerprint,
target_state_after_fingerprint}` is the concise structured trace the
taskblock asked for; full state dumps remain evidence-directory artifacts.

### Real accept fixture

Reused Slice 1/2's exact 26-token prompt. Oracle's own backbone forward
(24 blocks, `execute_qwen35_reference_token` looped once per prompt token)
was run for the first time end-to-end against the real 2B checkpoint
(previously Slice 2 only ran the MTP head with an externally-supplied
`h_nextn`) — this took ≈263s for 26 sequential scalar reference tokens
(≈10s/token), timed and budgeted before running, well within a single
evidence pass and no thermal concern.

```text
target_argmax_at_seed (Oracle's own backbone) = 248068
mtp_draft_argmax (Oracle's own MTP head)       = 248068
decision: accepted=true, canonical_token=248068
accounting: proposed=1, accepted=1, rejected=0, verification_count=1
```

Both values match Slice 1/2's beellama-derived record exactly.

### Real accept state equivalence

Lane A (ordinary AR: forward the target's own greedy token) and Lane B (MTP
proposes, Oracle verifies, accept, forward the canonical token) were run
from two independent copies of the identical post-prompt `HybridCache`.

```text
Lane A state: sequence_length=27 combined=0x97082dbd652cf8b3
Lane B state: sequence_length=27 combined=0x97082dbd652cf8b3
LANE_STATE_EQUAL: YES
LANE_NEXT_LOGITS_BITWISE_EQUAL: YES (both lanes' next-position logits are bit-identical,
                                      and both independently argmax to 271)
```

Canonical target state is bit-identical between the two lanes — full
geometry (all attention K/V, all recurrent convolution/recurrent state),
not merely sequence length.

### Real rejection search

Per the taskblock's explicit preference, the search used the pinned
GPU-capable beellama reference directly (not Oracle's scalar CPU path):
32 short, varied prompts (narrative/technical/list/open-ended, chosen for
higher local entropy than low-perplexity factual Q&A), up to 8 MTP
opportunities each, same per-step mechanics as the Slice 2 capture harness
(target `h_nextn`, unmasked, feeds a one-token MTP decode; comparison
against the target's own already-produced logits). **A natural disagreement
was found at the first opportunity of the 4th prompt**, so the search
stopped there per instruction ("stop at the first clean deterministic
disagreement"): `TOTAL_POSITIONS_SEARCHED=4`, `ALL_MATCH_COUNT=3`. (The
scan tool's per-prompt continuation past a first *matching* position had a
harness-side defect — documented transparently in the evidence log — that
limited effective exploration to one position for the three prompts checked
before the mismatch; this does not affect the mismatch finding itself,
which used the same single-step mechanism already validated by the accept
fixture.)

### Real reject fixture

```text
prompt: "The three ingredients you need are" (tokens 760,2250,13565,488,1144,513)
position: 5
target token (beellama):  25   |  target token (Oracle's own backbone): 25
MTP draft token (beellama): 279 | MTP draft token (Oracle's own MTP head): 279
decision: accepted=false, canonical_token=25
accounting: proposed=1, accepted=0, rejected=1, verification_count=1
```

Oracle's own values match beellama's found values exactly.

### Real reject state equivalence

Same Lane A/B construction as the accept case, forwarding token 25 in both
lanes (Lane A: the target's own choice; Lane B: the verification's
canonical token, which equals the target's choice because the draft was
rejected).

```text
Lane A state: sequence_length=7 combined=0x6932c8b39e9675a1
Lane B state: sequence_length=7 combined=0x6932c8b39e9675a1
LANE_STATE_EQUAL: YES
LANE_NEXT_LOGITS_BITWISE_EQUAL: YES
canonical_token (25) != rejected draft (279): confirmed -- zero residue from the
  discarded draft, by construction (verify_qwen35_mtp_proposal never touches state).
```

### Synthetic tests

`tests/test_phase2f_slice3.cpp` (new), 12 tests covering every item in the
taskblock's matrix: depth-1 accept; depth-1 reject with explicit target-authority
assertion (`canonical_token` differs from the discarded draft); canonical state
advances by exactly one position and commits exactly once (no double commit);
a rejected draft is provably invisible to a real `HybridCache` (`verify_qwen35_mtp_proposal`
takes no cache parameter, and the fingerprint before/after is asserted equal);
MTP's isolation extends through the full propose→verify sequence; accounting
exactness for both accept and reject; deterministic tie resolution (lower
index wins, consistently for draft and target); non-finite target/draft
logits rejected; malformed logits width, invalid vocabulary size, and
self-inconsistent draft token all rejected; draft/target position mismatch
rejected; bit-identical determinism across repeated verification; and
diagnostic-trace assembly (asserting the state fingerprint actually changes
once the canonical token is forwarded, and that the text/JSON renderers
surface the decision correctly).

### State isolation and accounting

`verify_qwen35_mtp_proposal`'s signature has no state parameter — the
strongest possible isolation guarantee, stronger than "an isolated state
object," because there is no object for canonical code to accidentally
alias. Every real and synthetic fixture confirms `drafts_proposed == 1`,
`drafts_accepted + drafts_rejected == 1`, `verification_count == 1`.

### Reference comparison summary

| field | beellama (real) | Oracle (real) | match |
|---|---|---|---|
| accept: draft token | 248068 | 248068 | yes |
| accept: target token | 248068 | 248068 | yes |
| accept: decision | accept | accept | yes |
| accept: canonical token | 248068 | 248068 | yes |
| reject: draft token | 279 | 279 | yes |
| reject: target token | 25 | 25 | yes |
| reject: decision | reject | reject | yes |
| reject: canonical token | 25 | 25 | yes |

Timing was not used as correctness evidence anywhere in this comparison.

## Deliberate exclusions (per task instruction)

- No speculative decoding beyond depth 1: no multi-token draft loop, no
  chained MTP predictions, no batched target verification, no partial
  multi-token acceptance, no adaptive draft depth, no acceptance-rate
  tuning.
- No rollback/checkpoint/snapshot machinery — Slice 3 proved depth-1
  correctness does not require it (see "Pinned depth-1 verification
  timeline" above) and none was added.
- Neither `execute_qwen35_reference_mtp` (Slice 2) nor
  `verify_qwen35_mtp_proposal` (Slice 3) is integrated into
  `Qwen35GenerationSession` or any generation path.
- No CUDA/Vulkan/SIMD/threading/scheduler work. No HTTP/API surface. No
  Lumina integration. No performance claims (the stage-by-stage divergence
  table in the Slice 2 section, and the timing figures in this section, are
  correctness/contract/evidence records, not benchmarks).
- `Qwen35MtpTrace`/`Qwen35MtpResult`'s "MTP block stages" are exactly
  Oracle's own reused `Qwen35BlockTrace` tensor names (`attn_norm`,
  `Qcur_full`, ...), not `graph_mtp`'s `cb()` names (`mtp_attn_norm`,
  `mtp_Qcur_full`, ...) — the evidence directory's comparison script maps
  between the two explicitly; no attempt was made to rename Oracle's
  existing, already-validated trace vocabulary to match beellama's.

## Slice 4 — Multi-draft MTP chaining and sequential partial-acceptance verification

### Pinned multi-draft chain contract

Source-verified from `beellama/common/speculative.cpp`,
`common_speculative_impl_draft_mtp::draft()` (lines 918-1041) and `::accept()`
(lines 1043-1056), commit `a620cbd481d16d3ccb5c8c96fa2fbd70191bea38`:

- The seed decode uses the anchor token (`dp.id_last`) at RoPE position
  `dp.n_past`, with `embd` set to the target's own `h_nextn` (its last hidden
  state, taken after the backbone's final `output_norm` — the same value
  Slice 2 established as the MTP block's input, never one of `graph_mtp`'s
  own outputs).
- Each subsequent draft `D_k` (`k` > 0) samples from the *previous* decode's
  logits, and its input hidden state is that *same* previous decode's own
  `t_h_nextn` output (the MTP block's post-`shared_head_norm` value,
  captured immediately before the tied output head is applied) — never the
  target's `h_nextn` again. `D_k`'s input token is `D_{k-1}`'s sampled id.
- The MTP block's own KV cache (`ctx_dft`) accumulates across the *whole*
  chain: one `llama_decode(ctx_dft, ...)` per step, never reset mid-chain,
  causally attending to every prior step in the same chain.
- `params.n_max` is a hard depth cutoff. `p_min` is a confidence-gating
  heuristic for early stopping — a policy decision, not an architectural
  requirement — and is explicitly excluded from Slice 4 ("no adaptive draft
  depth").

### Oracle chain generation API

`include/oracle/runtime/qwen35_mtp.hpp` was extended (not rewritten): the
Slice 2 execution core is factored into
`execute_qwen35_reference_mtp_step(manifest, weights, input, KvCache&
mtp_state, capture_intermediates)`, which accepts an *external*,
caller-owned `KvCache` instead of creating one internally, plus
`Qwen35MtpStepOutput{result, chained_hidden_state}` exposing the block's own
post-`shared_head_norm` value (needed for chaining, never surfaced by
Slice 2's `Qwen35MtpResult` since a single depth-1 prediction never needs
it). `execute_qwen35_reference_mtp` (Slice 2's original signature) is now a
one-line wrapper calling the step function with a fresh capacity-1
`KvCache` — verified regression-safe: rebuilt and re-ran the Slice 2
validation harness against the real checkpoint and got the *identical*
fingerprint (`0x175da26d295e89c1`, `TOP_20_OVERLAP 19/20`, `ARGMAX_MATCH
YES`) as before the refactor.

`include/oracle/runtime/qwen35_mtp_chain.hpp`'s
`generate_qwen35_mtp_draft_chain(manifest, weights, seed_token,
seed_hidden_state, seed_position, max_depth, capture_intermediates)` drives
`execute_qwen35_reference_mtp_step` in a loop over one shared, growing
`KvCache mtp_state(max_depth, ...)`, chaining `current_token`/`current_hidden`
from each step's `draft_token`/`chained_hidden_state` exactly as the pinned
`draft()` loop does. Rejects `max_depth == 0`. Each returned `Qwen35MtpDraft`
carries `draft_index`, `position` (the *predicted* canonical position,
`seed_position + index + 1`), `input_token`, `draft_token`, full `logits`,
and diagnostic-only fingerprints of its input hidden state and MTP KV state.

### Real depth-3 chain capture (beellama)

A disposable harness (`model-reports/.../slice4-chain/mtp_chain_capture.cpp`)
reproduces the pinned `draft()` loop directly against beellama's public API
(`llama_decode`/`llama_get_logits_ith`/`llama_get_embeddings_nextn_ith`),
using the established capital-of-France fixture (prompt token count 26,
seed position 25, seed token 198 — same fixture as Slices 1-3). Result:

| draft | RoPE pos | predicted pos | input token | draft token (argmax) |
|---|---|---|---|---|
| D0 | 25 | 26 | 198 | 248068 |
| D1 | 26 | 27 | 248068 | 271 |
| D2 | 27 | 28 | 271 | 248069 |

D1's input token (248068) is exactly D0's draft token, and D2's input token
(271) is exactly D1's draft token — proof of genuine chaining, not
independent re-seeding of the same anchor (the taskblock's explicit
caution, since Slice 1's `draft_n=9, draft_n_accepted=9` record does not by
itself prove any particular chain length). D0's own fingerprints
(`SEED_H_NEXTN_FNV1A64 0xb99610ab1cb34658`, `logits_fnv1a64
0xcce85f6074b0d7a3`, `draft_token 248068`) are bit-identical to Slice 2's
independently-captured single-step record, cross-validating both harnesses
against each other.

### Oracle chain comparison vs. real capture

`mtp_chain_validate.cpp` drives Oracle's chain two ways against the same
real capture: (a) a manual step loop calling
`execute_qwen35_reference_mtp_step` directly (byte-identical fingerprints to
the capture harness, so the numbers are directly comparable), and (b) the
public `generate_qwen35_mtp_draft_chain` API, confirmed to agree with (a) on
every `draft_index`/`position`/`input_token`/`draft_token`
(`CHAIN_API_MATCHES_MANUAL_LOOP YES`).

| draft | argmax match | logits max_abs | logits l2_rel | logits TOP-20 overlap | chained-h max_abs | chained-h l2_rel |
|---|---|---|---|---|---|---|
| D0 | YES (248068) | 2.71e-01 | 1.84e-02 | 19/20 | 5.03e-01 | 2.01e-02 |
| D1 | YES (271) | 3.79e-01 | 2.21e-02 | 19/20 | 3.88e-01 | 2.76e-02 |
| D2 | YES (248069) | 3.83e-01 | 2.46e-02 | 19/20 | 6.42e-01 | 2.83e-02 |

Argmax agreement holds at every step of the chain. The numeric divergence is
the same, already-explained decoded-F32-scalar-vs-production-quantized-kernel
difference Slice 2 established starting at the Q8_0 `eh_proj` matvec (D0's
figures here match Slice 2's original record exactly) — it accumulates mildly
across the chain (each step's small input perturbation compounds into the
next) but stays in the same explained range at every step, never blows up,
and never flips an argmax.

### Sequential target verification + partial acceptance API

`include/oracle/runtime/qwen35_mtp_chain_verify.hpp`'s
`verify_qwen35_mtp_draft_chain(manifest, weights, chain,
anchor_target_logits, HybridCache& canonical_state, configured_max_depth)`
is, unlike Slice 3's `verify_qwen35_mtp_proposal`, genuinely stateful: depth
> 1 verification requires a fresh target forward between drafts (there is
no way to obtain `D_1`'s comparison logits without first having forwarded
whatever canonical token resulted from `D_0`'s decision). After validating
the chain (non-empty, within `configured_max_depth`, contiguous positions,
matching logits width, valid token ids, finite logits), it walks the chain
in order: `target_token = argmax(current target logits)`; accept iff
`draft.draft_token == target_token`; either way, forward exactly the
resulting `canonical_token` through the target once
(`execute_qwen35_reference_token`), advancing `canonical_state` by exactly
one position; stop at the first rejection and discard the unused suffix. No
canonical speculative write ever happens before a decision, so no rollback
is used or needed — this is Slice 3's finding, reaffirmed unchanged for
chains. Accounting invariants (`Qwen35MtpChainAccounting`): `accepted +
rejected <= proposed`; `rejected` is 0 or 1 (first-rejection semantics);
`unused_suffix == proposed - accepted - rejected`; `full_chain_accepted <=>
(accepted == proposed && rejected == 0)`.

### Synthetic verification test matrix

`tests/test_phase2f_slice4.cpp` uses an all-zero-weight fixture (every
RMSNorm weight zero-initialized, so every norm output — and everything
downstream of it — is exactly zero regardless of input or history,
cascading to a deterministic real argmax of 0 at every position) to get
full, *honest* control over multi-step accept/reject scenarios, since
`verify_qwen35_mtp_draft_chain` performs genuine forwards and cannot be fed
mocked intermediate target logits. Covers, verbatim from the taskblock: A
(all accept), B (reject first), C (partial accept), D (reject middle with
suffix); depth zero rejected; depth above configured maximum rejected;
non-contiguous/invalid proposal positions rejected; invalid token ids
rejected; non-finite logits rejected; deterministic repeated execution; and
chain-generation depth 1/2/3 (confirming `D_k`'s input token equals
`D_{k-1}`'s draft token — proof the generator chains rather than
re-seeding). All pass.

### Real full-accept fixture

A second disposable harness
(`target_chain_verify_capture.cpp`) continues the *target* context (not the
MTP draft context) through the real depth-3 chain's draft tokens one at a
time, recording the target's own greedy argmax at each new position — the
real ground truth sequential verification compares against:

| position | fed token | target's own argmax | compares against |
|---|---|---|---|
| 25 (anchor) | — | 248068 | D0 (248068) — **accept** |
| 26 | 248068 | 271 | D1 (271) — **accept** |
| 27 | 271 | 248069 | D2 (248069) — **accept** |

All three drafts are accepted by the real target, confirming a genuine
depth-3 full-accept chain on the canonical fixture (distinct from, and more
rigorous than, assuming Slice 1's `draft_n=9` record implies any particular
chain shape).

### Real partial-reject fixture (bounded search, not found)

A bounded scan (`mtp_chain_rejection_scan.cpp`, extending Slice 3's
single-draft `mtp_rejection_scan.cpp` to depth > 1 chains) searched 33
short, varied prompts (one over the taskblock's 32-prompt guidance, due to
appending the canonical capital-of-France prompt for fixture consistency —
noted as a deviation below), up to 8 chain-start positions per prompt, depth
≤ 3, for a chain where ≥ 1 draft is accepted and a later draft in the *same*
chain is rejected. Result: **not found** within the bound (66 total
chain-start attempts searched, `FOUND NO`). Full search evidence is archived
in `slice4-chain/real-partial-reject-chain-scan.log`, per the taskblock's
explicit allowance for a bounded, honestly-reported not-found outcome. The
zero-accept case remains covered by Slice 3's reused real fixture ("The
three ingredients you need are", position 5, target 25, draft 279), and
partial-accept-then-reject is covered by the synthetic matrix (items C, D)
plus the Lane A/B equivalence check below.

### Canonical state equivalence, Lane A/B (chained cases)

Extends Slice 3's single-draft Lane A/B finding to multi-draft chains.
`test_canonical_state_equivalence_chained_full_accept` and
`test_canonical_state_equivalence_chained_partial_reject` (added to
`tests/test_phase2f_slice4.cpp`) each build two independent `HybridCache`
instances seeded identically: Lane A runs `verify_qwen35_mtp_draft_chain`
directly; Lane B is never touched by the verifier at all — it is manually
forwarded, token by token, through the exact `canonical_tokens` sequence
Lane A's own accounting returned, calling
`execute_qwen35_reference_token` directly. Both lanes' final states are
compared via `fingerprint_qwen35_state` (Slice 3's diagnostic FNV1a64
state fingerprint, covering every attention block's K/V and every recurrent
block's convolution/recurrent state). Both the full-accept (3/3 drafts) and
partial-reject (1 accepted, 1 rejected, 2 unused-suffix discarded) cases
produce bit-identical Lane A/B fingerprints, proving chain verification does
nothing to canonical state beyond "forward each decided token once, in
order" — no residue from rejected or unused-suffix drafts ever reaches
canonical state.

### Build gates and regression (Slice 4)

Fresh GCC Release, fresh Clang Release (both zero warnings), and
Clang ASan+UBSan+leak-detection builds all pass 15/15 tests
(`slice4-gcc-ctest.log`, `slice4-clang-ctest.log`,
`slice4-sanitize-ctest.log`). The canonical Phase 2D standalone fingerprint,
`0x54825e50fa9398cf`, was re-verified bit-identical against the fresh Slice
4 GCC build. `src/runtime/qwen35_generation.cpp` has a zero-line diff
against the pinned base commit (`git diff
162f0451bd4fd9710d0cc401fd4a6e4ed044c7df -- src/runtime/qwen35_generation.cpp`)
and `oracle-phase2e-tests` passes on every Slice 4 build — Phase 2E is
unmodified.

### Clean reconstruction (Slices 1+1B+2+3+4)

A fresh `git worktree add --detach` at
`162f0451bd4fd9710d0cc401fd4a6e4ed044c7df`, with the full cumulative diff
applied as a patch and every new (untracked) file copied in by hand,
reproduced a byte-identical diff against
`162f0451bd4fd9710d0cc401fd4a6e4ed044c7df` (`diff <(git diff ...)
<(git diff ...)` — zero lines of output) to the working tree. All three
build gates (GCC/Clang/sanitize) and the Phase 2D fingerprint check were
re-run from scratch in that clean worktree with identical results: zero
warnings, 15/15 tests, `0x54825e50fa9398cf`.

### Deviations / open questions

- The partial-reject bounded scan used 33 prompts, one over the taskblock's
  "≤32 prompts" guidance (32 varied prompts plus the canonical
  capital-of-France prompt, appended for fixture-set consistency with the
  rest of Slice 4's evidence). This is a strict superset of a 32-prompt
  search and was run to completion before being reported as not-found; it
  is noted here for exactness rather than silently rounding down to "32."
- No real partial-accept-then-reject chain was found within the bound; this
  scenario is covered by the synthetic test matrix and the Lane A/B
  equivalence check instead, as the taskblock explicitly allows.
- The clean-reconstruction worktree for this slice was created under the
  session scratchpad directory rather than a persistent `/home/bino/`
  sibling directory (the convention used by Slices 1-3's clean-reconstruction
  worktrees). This is a path-location difference only; the reconstruction
  itself (diff equivalence, build gates, fingerprint) is identical in kind
  and result to prior slices'.

### Deliberate exclusions (per task instruction)

- No batched target verification (each draft is verified against a fresh,
  individually-produced target forward — never a single batched decode
  covering the whole chain).
- No target rollback or speculative pre-write of target KV — state only
  ever advances forward, one decided token at a time (Slice 3's finding,
  reaffirmed for chains: the pinned reference's own rollback exists only
  because it speculatively pre-writes target KV before the decision, a
  performance strategy Oracle's decide-first architecture has no need for).
- No bonus-token optimization, no adaptive draft depth (`p_min` early-stop
  is explicitly excluded, matching the pinned contract section above).
- Not integrated into `Qwen35GenerationSession`. No CUDA/SIMD/threading/
  scheduler/HTTP/Lumina integration. No performance tuning or claims — all
  numeric comparisons above are correctness/contract evidence, not
  benchmarks.

### Verdict

`VERDICT=PHASE2F_SLICE4_MULTI_DRAFT_PARTIAL_ACCEPTANCE_MATCH`

## Slice 5 — Batched target verification and canonical prefix rollback

### Pinned batched verification contract

Source-verified from `beellama/tools/server/server-context.cpp`'s
`use_mtp_spec_accept` block (~line 6935) and `slot::update_batch` (~line
1352), plus `common/sampling.cpp`'s `common_sampler_sample_and_accept_n`
(~line 772), commit `a620cbd481d16d3ccb5c8c96fa2fbd70191bea38`:

- **Batch construction**: one target batch holds the anchor token (`slot.sampled`,
  already canonical) at position `pos0 = prompt.tokens.pos_next()`, followed by
  every draft token at `pos0+1, pos0+2, ...`. `spec_i_batch[0]` records the row
  index of the *anchor's* logits; `spec_i_batch[i+1]` records the row index of
  `draft[i]`'s logits.
- **Verification logit alignment (the one-token shift)**: `draft[0]` is verified
  against `spec_i_batch[0]`'s logits — the *anchor's own* row, produced before any
  draft token was ever decoded. `draft[i>0]` is verified against `spec_i_batch[i]`'s
  logits — the row produced when the batch decoded `draft[i-1]`. This is exactly
  the same one-token shift Slice 3/4 already established for sequential
  verification, just computed from one batched decode instead of N sequential ones.
- **Accepted-prefix determination**: `common_sampler_sample_and_accept_n` walks
  `draft[0..n-1]` in order, sampling from each verification row; it stops at the
  first `draft[i] != sampled`. If every draft matches, it samples one *additional*
  token from `spec_i_batch[n]`'s row (the row produced by decoding the last draft
  token) — the "bonus" token (see below).
- **Speculative target KV**: written for the *entire* batch (anchor + all drafts)
  before any acceptance decision is made — this is what makes it "batched": the
  target model does not know which drafts will be rejected when it writes their KV.
- **On rejection** (`n_rollback > 0`): if the underlying KV memory module supports
  bounded partial `seq_rm` (`COMMON_CONTEXT_SEQ_RM_TYPE_PART`/`_RS`), positions from
  the new canonical boundary onward are removed directly. If not (`_FULL`, or an
  `_RS` rollback deeper than the module's bounded `n_rs_seq` snapshot window allows),
  the *entire* speculative batch's target/draft state is restored from a checkpoint
  taken *before* the batch was decoded, and the confirmed-accepted prefix is
  resubmitted as a new draft to be recommitted on the *next* iteration.
- **On full acceptance**: no rollback at all — every speculative write was correct,
  and the bonus token is folded directly into canonical output.
- **Bonus token**: source-verified to be a pure optimization artifact, not a
  distinct sampling semantic. It is produced by `common_sampler_sample_and_accept_n`
  reading the logits row that decoding the *last accepted draft* already produced
  as part of the batch — i.e. exactly the value ordinary next-step decoding would
  compute anyway; batching just makes it available one step early "for free."

### Verification logit alignment

Proven two ways. First, directly from source (above): `draft[i]`'s verification
row is `spec_i_batch[i]` (anchor row for `i=0`, else the row produced by decoding
`draft[i-1]`) — implemented identically in
`verify_qwen35_mtp_draft_chain_batched` (`src/runtime/qwen35_mtp_chain_verify_batched.cpp`):
`draft[0]` is checked against the caller-supplied `anchor_target_logits`; `draft[i>0]`
against `speculative.steps[i-1].logits`, where `speculative` is the result of
one `execute_qwen35_reference_target_multi` call over the whole draft-token
sequence. Second, empirically, via `tests/test_phase2f_slice5.cpp`'s
`test_verification_logit_alignment`: a dedicated *identity fixture* (every block's
output/down projection is zero, so every block passes its input straight through
via the residual; `token_embd` is one-hot per token and `output_norm.weight` is
uniform, so the tied output head makes decoding token `T` always argmax to `T`
itself) gives genuinely input-dependent, distinct verification logits at every
position — unlike the all-zero fixture used elsewhere, which is constant
everywhere and could mask a misaligned row mapping. A chain is constructed where
an off-by-one row mapping would flip `draft[2]`'s accept/reject decision; both
lanes are asserted to reject at exactly index 2, with matching canonical tokens
and final state fingerprints.

### Speculative target state design

Section 7 of the taskblock permits Slice 5 to temporarily write draft tokens into
the *target's own* `HybridCache` before they are known to be canonical (unlike
Slice 4, which never does). `verify_qwen35_mtp_draft_chain_batched` does exactly
this: the whole chain is fed through `execute_qwen35_reference_target_multi`,
mutating `canonical_state` directly and immediately — the "speculative" and
"canonical" cache are the *same object*, distinguished only by explicit
bookkeeping (a `HybridCacheBoundary` marked before the write, and three labelled
fingerprints in the result: `target_state_before_fingerprint`,
`target_speculative_state_fingerprint`, `target_state_after_fingerprint`). MTP's
own `KvCache` (Slice 2-4) is never touched by any of this — it remains a
completely separate object, used only during draft *generation*, never during
target *verification*.

### Recurrent state recovery

The highest-risk question in this slice (taskblock section 9/26). Source-proven
from `beellama/src/llama-memory-recurrent.cpp`: recurrent state cannot be
truncated by removing rows — the code's own comment states it plainly ("models
like Mamba or RWKV can't have a state partially erased at the end of the
sequence because their state isn't preserved for previous tokens"). Production's
own workaround is `n_rs_seq` ("number of recurrent-state snapshots per seq for
rollback"): a small, *bounded* ring of per-token recurrent-state snapshots,
letting `seq_rm` roll back up to `n_rs_seq` steps by switching to an earlier
snapshot slot (`set_rs_idx`) rather than mathematically undoing the recurrence;
rolling back further than the ring's depth is impossible and falls back to a full
checkpoint restore (confirmed by the `use_ckpt_tgt` branch in
`server-context.cpp`, taken whenever `n_rollback > llama_n_rs_seq(ctx)`).

Oracle's own `SsmState` (`include/oracle/runtime/hybrid_cache.hpp`) has the
identical constraint: `recurrent_` and `convolution_history_` are single,
fixed-size buffers mutated strictly in place (`reference_kernels.cpp`'s
`gated_delta_step` decays and accumulates into `recurrent_` in place;
`causal_depthwise_convolution_step` shifts `convolution_history_` in place) —
confirmed by direct inspection, not assumed. There is no per-token history to
truncate back to.

Given this, and given production's own necessary fallback to full snapshot
restore for exactly this case, Slice 5 uses a **single-boundary snapshot**,
narrower than production's N-deep ring (Oracle's speculative pattern only ever
needs to recover to *one* point — the canonical position immediately before a
speculative batch — never to an arbitrary intermediate depth):
`HybridCache::mark_boundary()` captures `sequence_length()` plus a full copy of
every SSM block's `recurrent_`/`convolution_history_` buffers (small,
fixed-size; attention blocks need no data captured at all, see below).
`HybridCache::truncate_to(boundary)` restores every SSM block's buffers and
`sequence_length_` from that one boundary, and validates that it is never asked
to grow state or restore a shape-mismatched boundary. This is not a general
snapshot/restore framework — there is no stack, no arbitrary depth, no way to
mark more than one boundary's worth of recurrent history at a time — and no
STOP was required: a narrow, exact, source-precedented strategy was available.

### Attention state recovery

Trivial by comparison and requires no snapshot at all: Oracle's `KvCache` stores
one K/V row per token position, appended forward-only (`KvCache::append`), and
`append` only ever writes at the *current* length. Rows below any given
`new_length` have therefore never been overwritten and remain bit-identical
regardless of what was appended past them — `KvCache::truncate(new_length)` is
just `length_ = new_length`, exact and lossless by construction. Verified
directly in `test_hybrid_cache_truncate_to_direct`: the byte contents of the
retained prefix's first attention row are asserted identical before and after
truncating away later positions.

### Oracle batched verification API

- **`execute_qwen35_reference_target_multi`** (`include/oracle/runtime/qwen35_target_multi.hpp`):
  executes N known tokens through the target at consecutive positions, retaining
  each step's logits, advancing `state` exactly as ordinary generation would.
  Internally a loop over the already-validated `execute_qwen35_reference_token` —
  Oracle has no batched compute backend, so this models the pinned reference's
  *batch contract* (one ordered set of verification logits per token) without any
  batched-performance implication, none of which is claimed anywhere in this
  slice.
- **`verify_qwen35_mtp_draft_chain_batched`** (`include/oracle/runtime/qwen35_mtp_chain_verify_batched.hpp`):
  an independent implementation of Slice 4's contract — never calls, wraps, or
  routes through `verify_qwen35_mtp_draft_chain`. Algorithm: mark a boundary;
  speculatively evaluate the *whole* chain via `execute_qwen35_reference_target_multi`;
  determine the accepted prefix from that pass's own logits (draft[0] vs the
  caller-supplied anchor logits, draft[i>0] vs the previous step's real output);
  if fully accepted, keep the speculative write as-is (nothing to undo); otherwise
  `truncate_to(boundary)` and recommit *exactly* the confirmed canonical sequence
  (accepted drafts plus the one rejection-correcting token) via a second, short
  pass. The second pass deliberately recomputes the (already-correct) accepted
  prefix's attention KV rather than trying to preserve it in place — a simplicity/
  correctness tradeoff, not a performance claim (see "target state commit model").
  Returns the same accounting surface as Slice 4's `Qwen35MtpChainAccounting`
  (independently produced, never copied), plus `speculative_target_tokens_evaluated`,
  the three labelled state fingerprints, and `final_target_logits`.

### Target state commit model

Option **A** from the taskblock (mutate canonical state speculatively, then
roll back on rejection) — chosen because Oracle's target forward path has no
notion of a "shadow" state distinct from the real `HybridCache` (unlike
`llama.cpp`'s discrete backend buffers, there is nothing cheaper to write
speculatively into), and because attention rows are naturally, losslessly
undoable by construction (no cost to speculating on them) while recurrent state
is cheap to snapshot once given its small, fixed size. State is always in
exactly one of three explicitly observable conditions, verified via the result's
three fingerprint fields: **canonical** (`target_state_before_fingerprint`, prior
to any speculative write), **speculative** (`target_speculative_state_fingerprint`,
immediately after the whole chain is spec-evaluated — this may include rejected
or unused-suffix drafts that never become canonical), and **committed**
(`target_state_after_fingerprint`, after either keeping the speculative write in
full or truncating and recommitting only the confirmed prefix). No hidden
mutation: every state transition is bracketed by an explicit fingerprint the
caller can observe and compare.

### Synthetic Lane A/B comparisons

`tests/test_phase2f_slice5.cpp` reruns Slice 4's exact A/B/C/D matrix through
`compare_lanes`, which independently seeds two `HybridCache` instances, runs
Slice 4's sequential verifier on one and Slice 5's batched verifier on the
other, and asserts full equality of every accounting field, final
`sequence_length()`, the complete `fingerprint_qwen35_state` result, and (via an
independent dummy-forward "peek" on a copy of each lane's final state) the
next-target logits. All four cases pass, plus the dedicated alignment test
above and the full rollback-specific matrix below.

### Real full-accept fixture

Oracle's own tokenizer, forward path, and `generate_qwen35_mtp_draft_chain`
reproduce the established depth-3 capital-of-France chain
(`D0=248068, D1=271, D2=248069`) end to end, natively. Both lanes report
`proposed=3 accepted=3 rejected=0 full_chain_accepted=true`, identical
`canonical_tokens=[248068,271,248069]`, identical `sequence_length()=29`, and a
**bit-identical** final `HybridCache` fingerprint
(`0x9bbd2c1aeb69f4c2` on both lanes).

### Real zero-accept fixture

Re-derived independently through Oracle's own path (not loaded from a prior
slice's record): `"The three ingredients you need are"`, seed position 5, seed
token 513, target argmax 25, MTP draft argmax 279 — bit-for-bit the same values
Slice 3 established from beellama, now reproduced natively end to end. Both
lanes reject at index 0, both produce `canonical_tokens=[25]`,
`sequence_length()` lands at exactly `length_before + 1` on both lanes (the
speculative suffix beyond the rejection point never survives), and the final
fingerprints match. A **third**, independent cache built via plain ordinary
autoregressive forwarding (prompt, then canonical token 25, no speculation
involved at all) was also fingerprinted and found **bit-identical** to both
speculative lanes' post-rollback state — the strongest available evidence that
rollback leaves the target completely indistinguishable from as if speculation
had never been attempted.

### Real partial-reject fixture

Not pursued: Slice 4 already ran a bounded natural-partial-reject search (33
prompts × ≤8 chain starts × depth ≤3) and reported not-found; per this
taskblock's explicit instruction ("Do NOT rerun a large discovery search unless
needed... synthetic partial rejection is sufficient"), no new search was run.
Partial rejection is covered by the synthetic matrix (cases C/D) and the
dedicated rollback tests below, both of which exercise the real target forward
path (the all-zero-weight fixture), just not real English-language draft/target
disagreement.

### Bonus token forensics

Diagnostic only, never integrated into generation. On the real full-accept
fixture, `verify_qwen35_mtp_draft_chain_batched`'s `final_target_logits`
(available "for free" from the already-decoded last draft token, matching the
pinned reference's own bonus-token mechanics) argmax to token 271 at position
29, fingerprint `0xc5446b0f5c14e05a`. Source forensics (above) already
established the bonus token is a pure optimization artifact, not a distinct
sampling semantic — the same value ordinary next-step decoding would compute
regardless of whether the batch happened to expose it early. This is recorded
as evidence only; no bonus token is ever folded into canonical generation
output in this slice.

### Canonical state equivalence / accounting

Every Lane A/B comparison above (synthetic A-D, the alignment test, the six
rollback-specific tests, and both real fixtures) asserts the identical
accounting surface Slice 4 defines (`proposed`, `accepted`, `rejected`,
`unused_suffix`, `verification_count`, `first_rejection_index`,
`full_chain_accepted`, `canonical_tokens`) plus full `HybridCache` state
fingerprint equality. `accepted + rejected <= proposed`, `rejected` is 0 or 1,
and `unused_suffix == proposed - accepted - rejected` hold identically to
Slice 4 in every case (produced independently, not inherited).

### Rollback-specific coverage

`tests/test_phase2f_slice5.cpp`: rollback after zero/one/multiple accepted drafts
(each asserting the exact surviving `sequence_length()` and that the speculative
and after-commit fingerprints differ, proving a real rollback occurred); no
rollback required on full acceptance (speculative and after-commit fingerprints
asserted *identical*, proving `truncate_to` never ran); `HybridCache::truncate_to`
direct coverage — no-op at current length, rejection of a future-length boundary
("must not grow"), rejection of a shape-mismatched boundary (different SSM layer
count), byte-identical retained attention prefix, and deterministic repeated
truncation to the same boundary; and unused-suffix residue (`sequence_length()`
after a reject-middle-with-suffix run equals exactly `length_before + accepted +
rejected`, never including any of the discarded unused-suffix drafts).

### Compiler/sanitizer gates

Fresh GCC Release, fresh Clang Release (zero warnings on both), and Clang
ASan+UBSan+leak-detection all pass 16/16 tests
(`slice5-gcc-ctest.log`, `slice5-clang-ctest.log`, `slice5-sanitize-ctest.log`).

### Phase 2D regression

Canonical fingerprint `0x54825e50fa9398cf` re-verified bit-identical against the
fresh Slice 5 GCC build.

### Phase 2E regression

`src/runtime/qwen35_generation.cpp` has a zero-line diff against
`162f0451bd4fd9710d0cc401fd4a6e4ed044c7df`; `oracle-phase2e-tests` passes on
every Slice 5 build. Slice 5 remains entirely outside `Qwen35GenerationSession`.

### Clean reconstruction (Slices 1+1B+2+3+4+5)

Fresh `git worktree add --detach` at `162f0451bd4fd9710d0cc401fd4a6e4ed044c7df`,
cumulative diff applied as a patch, every new file copied in by hand, diff
verified byte-identical to the working tree, all three build gates and the
Phase 2D fingerprint re-run from scratch with identical results.

### Deviations / open questions

- The batched verifier's rejection-path "commit" pass recomputes the
  already-correct accepted prefix's attention KV rather than preserving it in
  place (which would require the same N-deep recurrent snapshot ring production
  uses, explicitly out of scope: "no performance policy"). This is a deliberate
  simplicity/correctness tradeoff, not a limitation of the rollback mechanism
  itself — no production-performance claim is made anywhere in this slice.
- The identity fixture used for the alignment-proof test requires
  `embedding_length == vocabulary_size` (a one-hot-per-token construction); this
  is a test-only fixture choice with no bearing on the production Qwen3.5
  manifest, which never has this property.

### Deliberate exclusions (per task instruction)

- No integration into `Qwen35GenerationSession`, no user-facing speculative
  generation, no adaptive draft depth, no stochastic speculative sampling, no
  acceptance-rate tuning, no performance policy or claims.
- No CUDA/SIMD/CPU-threading/scheduler/HTTP/API/Lumina integration, no Tiered
  Weight Residency.
- Slice 4's sequential verifier (`verify_qwen35_mtp_draft_chain`) is unmodified,
  un-weakened, and never called from or routed through the batched
  implementation — confirmed by inspection: `qwen35_mtp_chain_verify_batched.cpp`
  has no dependency on `qwen35_mtp_chain_verify.hpp`/`.cpp` at all.
- The bonus token is captured as diagnostic evidence only and is never folded
  into canonical generation output.

### Verdict

`VERDICT=PHASE2F_SLICE5_BATCHED_VERIFICATION_ROLLBACK_MATCH`

## Slice 6 — Guarded generation-session integration and MTP ON/OFF canonical parity

### Generation integration design

Read the complete landed Phase 2E implementation
(`include/oracle/runtime/qwen35_generation.hpp`,
`src/runtime/qwen35_generation.cpp`) before any change. `generate_fresh`'s
per-token commit body — real forward, real greedy sample, ledger append,
UTF-8 assembly, stop/reasoning/holdback bookkeeping, callback delivery,
termination check — is intricate and, by design, has two separate loop
implementations (a fast path with no stops/reasoning configured, and a slow
path with the Slice 3B/3C bounded-holdback machinery). Rather than
restructure either loop, Slice 6 makes exactly one narrow change, applied
identically to both: the single line `sampler.sample(current.logits)` is
replaced by a call to a small lambda, `obtain_next_token`, defined once and
shared by both loops. Confirmed by inspection of the diff against
`10fefb345f66b0fdc13e64798a56d3ef83c4d36d`: only **three lines** of existing
code were changed (two `sampler.sample(...)` call sites, and one JSON
closing brace to splice in the new diagnostics field) — everything else is
pure addition. No second, speculative-specific implementation of any
existing rule was written.

`obtain_next_token`'s own body still calls `sampler.sample(current.logits)`
— the exact same call, every single time, MTP on or off. When MTP is
enabled and no prediction is queued, it first proposes and shadow-verifies a
bounded draft chain (Slices 2-5, entirely unmodified) and queues the
resulting `canonical_tokens` as *predictions*; it then asserts that the
real, independently-derived greedy sample agrees with the head of that
queue. Because Oracle's forward is a deterministic scalar function of state
and MTP reference mode requires greedy sampling (enforced at preflight),
that agreement is guaranteed whenever the shadow state genuinely mirrored
canonical state — which it always does, since the shadow is a plain copy
taken immediately beforehand. A mismatch is therefore treated as an internal
consistency failure (`std::runtime_error`), never a recoverable "correction"
path. This is the entire mechanism by which MTP is structurally incapable of
changing canonical output: **MTP predicts; it never decides.**

### MTP configuration/API

`Qwen35MtpMode` (`disabled` default, `reference`) and
`Qwen35MtpGenerationConfig{mode, max_draft_depth}` are new fields on
`Qwen35GenerationRequest::mtp{}` — a default-constructed request is
unchanged in every way that matters to behavior, so every existing caller
compiles and behaves identically. Preflight (`validate_request`) rejects
misuse clearly, before any session state is touched: `mode != disabled` on a
manifest without `has_mtp()` throws immediately ("no MTP block") rather than
silently no-op'ing; `max_draft_depth` outside `{1,2,3}` throws; non-zero
`sampling.temperature` under MTP mode throws ("requires greedy sampling").

### Seed / h_nextn ownership

`current` (the `Qwen35ForwardResult` from whichever token was most recently
forwarded — the last prompt token, or the last committed generated token)
already carries everything Slice 2/4/5's contract requires as the MTP seed:
`current.logits` (anchor target logits), `current.trace.final_norm` (the
exact final-normalized h_nextn Slice 2 established — populated
unconditionally by `execute_qwen35_reference_token` regardless of the
`capture_block_outputs` flag Phase 2E already passes as `false`, so this
costs MTP-off callers nothing new), and `current.trace.token_id` (the anchor
token itself). No new storage was added to the session; `state_.sequence_length() - 1`
is the seed position.

### Shadow verification design

`HybridCache` is copy-constructible with no special handling required (a
plain value type over `std::vector<float>`s) — no STOP was needed for
section 4's "if HybridCache cannot be copied/shadowed narrowly" clause.
`obtain_next_token` takes `HybridCache shadow_state = state_;` immediately
before generating and verifying a draft chain via Slice 5's
`verify_qwen35_mtp_draft_chain_batched`, and lets `shadow_state` go out of
scope the moment the accounting is read — the live `state_` is never
touched during proposal/verification.

### Canonical commit path

Every canonical decision, whether it originated as an accepted draft or a
verification-time correction, is committed by popping one token id off the
prediction queue and letting the *existing* per-token body run exactly as
it always has: real forward, real sample (with the queue-agreement
assertion), ledger append, assembler append, stop/reasoning bookkeeping,
callback, termination check. If termination fires while committing token K
of a verified batch of size N, the loop breaks — as it always did — and the
remaining N-K predictions are simply never popped, never asserted against,
never emitted as events. No special "discard the rest of the batch" code
exists because there was never a batch-shaped code path to begin with: MTP
only ever changes how many times, in a row, the ordinary single-token loop
body runs before a fresh chain is proposed.

### Termination semantics / EOS / token-stop / text-stop parity

Precedence (EOS > token stop > text stop > reasoning-loop > max_tokens, with
context capacity checked before the next token) is verified unchanged
against the landed source before editing (section 11) and is untouched by
this slice — the termination checks live entirely inside the per-token body,
which MTP never modifies or bypasses. `test_eos_inside_verified_prefix`,
`test_token_stop_inside_verified_prefix`, and
`test_text_stop_inside_verified_prefix` each verify, on a real forward
(not a mock), that a >1-token MTP-verified batch correctly stops mid-batch
and discards the verified-but-uncommitted remainder, with full MTP
ON/OFF parity (tokens, text, stop_match, final `HybridCache` fingerprint).

### UTF-8 parity

Proven two ways: structurally (the assembler is invoked once per committed
token, in commit order, regardless of MTP — there is no code path that could
treat a whole verified batch as one fragment), and empirically, via a
dedicated real-forward "identity" fixture (one-hot `token_embd`, `eh_proj`
restricted to an exact identity submatrix on the e_norm/candidate-token half
of the concat — see "MTP-identity construction" below) driven through the
shared byte-fallback tokenizer fixture (`test_phase2e.cpp`'s
`write_tokenizer_fixture` pattern, reused verbatim): a token whose own raw
bytes are already a complete 2-byte UTF-8 character, repeated via a genuine
full-accept MTP chain, and a lone multi-byte lead byte repeated (which is
*not* valid on repetition), confirming both valid decode and U+FFFD
replacement are unaffected by MTP.

### Reasoning boundary / reasoning-loop / force-close parity

All three (ordinary boundary crossing, policy `stop`, policy `force_close`)
are exercised on a real forward via the same "MTP-identity" fixture, with
full MTP ON/OFF parity required and confirmed. Force-close in particular is
structurally untouched by MTP: the forced end-token sequence is injected by
literally the same code that always injected it (`execute_qwen35_reference_token`
called directly inside the reasoning-safety block), a code path
`obtain_next_token` is never invoked from — so section 18's "canonical
force-close behavior must remain a real target-model operation, never
MTP-synthesized" holds by construction, not by a special case.

### max_tokens / context-capacity parity (draft depth bounding)

`effective_depth = min(requested_depth, remaining_max_tokens_budget, remaining_context_capacity)`,
computed fresh at every draft opportunity. Both bounding quantities are
provably `>= 1` whenever `obtain_next_token` proposes at all (the loop's own
existing top-of-iteration `context_exhausted` check, and the previous
iteration's own `max_tokens` check, already guarantee this), so a depth
never gets proposed that a known hard boundary already forbids. Verified
with `remaining_budget` at 1 and 2 against a requested depth of 3, and with
a near-context-limit fixture (capacity 4, prompt occupying 1) — final
generated-token count and final sequence length never exceed the
configured limits, matching MTP OFF exactly.

### Callback / event parity

`compare_off_on` (the shared Lane A/B helper used by nearly every synthetic
test) captures every delivered callback event under both MTP OFF and MTP ON
and asserts identical count, order, token id, generated_index, text_fragment,
token_bytes, and eos flag. Rejected/unused drafts are confirmed to produce
*zero* events: `test_zero_accept_decision_batch` captures every event from a
run whose MTP chain natrually proposes tokens 8/19/26 (all rejected/unused)
and asserts every single delivered event's token id is the one real,
committed value (0) — the rejected draft ids never appear.

### MTP diagnostic accounting

`Qwen35MtpDiagnostics` (`mtp_enabled`, `requested_draft_depth`,
`draft_opportunities`, `drafts_proposed/accepted/rejected`, `unused_drafts`,
`verification_count`, `full_accept_chains`, `partial_accept_chains`,
`zero_accept_chains`) is attached to `Qwen35GenerationResult` and surfaced in
the text/JSON formatters. Purely observational — accumulated from each
opportunity's `Qwen35MtpChainAccountingBatched`, never read by the sampling
or commit path. Invariant tested directly: `full_accept + partial_accept +
zero_accept == draft_opportunities`, and `accepted + rejected <= proposed`
summed across the whole run.

### Synthetic test matrix

`tests/test_phase2f_slice6.cpp` covers all 25 items from the taskblock's
matrix. Three real-forward fixtures, each empirically validated (not
hand-derived) via a disposable probe before being written into the suite —
mirroring this project's established methodology
(`test_phase2e.cpp`'s `discover_natural_tokens`, Slice 3's bounded rejection
scan, Slice 4's chain rejection scan):

- **all-zero** (backbone and MTP both all-zero): every real forward
  argmaxes to token 0 regardless of input, and MTP's own `eh_proj` is zero
  too, so both sides agree on 0 forever — a genuine, real-forward-verified
  **full accept** at any depth, any prompt.
- **identity** (one-hot `token_embd`, uniform `output_norm`, zero
  `attn_output`/`ffn_down` everywhere for pure residual passthrough; MTP's
  `eh_proj` is an exact identity submatrix on the e_norm half, zero on the
  h_norm half — flat index verified against
  `reference_mapped_tensor_matvec`'s actual row-major convention in
  `quantized_reference.cpp`, `j*(2n)+j = 1`): decoding token T always
  argmaxes to T, and MTP always predicts its own candidate token, so a
  chained draft is `[T,T,T,...]` for *any* T — a genuine **full accept** for
  any real token id, used to drive the UTF-8/reasoning-boundary tests with
  real vocabulary entries.
- **textured** (identity-shaped embedding/norm, but MTP's `eh_proj` is a
  small fixed deterministic — not random — nonzero pattern): empirically
  searched across all 32 seed tokens at depth 3 and found to give a genuine
  **zero accept** for most seeds (e.g. 0) and a genuine **partial accept**
  (accepted=1, rejected=1, unused=1) at seed 5.

### Real MTP ON/OFF fixture

The established capital-of-France prompt (26 tokens), run through
`Qwen35GenerationSession` on the real 2B MTP checkpoint, 5 tokens, MTP OFF
vs MTP ON depth 3: **both produce the identical token sequence**
`[248068, 271, 248069, 271, 760]` (`<think>\n\n</think>\n\nThe`) — exactly
reproducing the established real depth-3 chain (`248068, 271, 248069`) as
genuine ordinary greedy continuation, plus two further tokens. Identical
`finish_reason` (`max_tokens`), identical `generated_text`, identical
`final_sequence_length` (31), and a **bit-identical** final `HybridCache`
fingerprint (`0xbbd63c415e657f3e`) on both lanes. MTP diagnostics for the ON
lane: 2 draft opportunities (depth 3 then depth 2, clipped by the remaining
budget), 5/5 drafts accepted, 2 full-accept chains, 0 rejected.

### Real rejection integration fixture

`"The three ingredients you need are"` (the established Slice 3/5 real
fixture) applies directly to `Qwen35GenerationSession` — `prompt_tokens` is
a plain token-id vector with no chat-template requirement, so no session
distortion was needed. MTP OFF and MTP ON both produce the identical
sequence `[25, 198]`, matching the established real target value (25) at
this position exactly, with a bit-identical final fingerprint. The MTP ON
lane's diagnostics for this prompt show genuine, real, organic accept/
reject/unused activity (3 proposed, 1 accepted, 1 rejected, 1 unused) at the
first opportunity — real evidence of partial-decision-batch dynamics
occurring naturally within a full session run on the real checkpoint, not
just at the lower-level Slice 4/5 API.

### Phase 2E regression

`git diff 10fefb345f66b0fdc13e64798a56d3ef83c4d36d -- src/runtime/qwen35_generation.cpp`
shows exactly 3 removed/changed lines (both documented above); every
existing Phase 2E synthetic test (`oracle-phase2e-tests`) passes unchanged.
The expensive real 4B (`Qwopus3.5`) `oracle-phase2e-slice2-fixture-v1`
matrix was **not** re-run, per the taskblock's explicit instruction ("do not
rerun the multi-day 4B matrix unless necessary... use the 2B MTP model for
new real integration evidence") — MTP-disabled behavior is proven unchanged
by the near-zero diff plus the full synthetic suite, and new real evidence
was gathered on the 2B MTP model instead (above).

### Phase 2D regression

Canonical fingerprint `0x54825e50fa9398cf` re-verified bit-identical against
the fresh Slice 6 GCC build.

### Slice 4 / Slice 5 regressions

Both suites (`oracle-phase2f-slice4-tests`, `oracle-phase2f-slice5-tests`)
rerun unchanged and green on every Slice 6 build. Neither is routed through
the other, and `verify_qwen35_mtp_draft_chain_batched` is called directly by
Slice 6 exactly as Slice 5 exposed it — no wrapper, no behavior change.

### Compiler/sanitizer gates

Fresh GCC Release, fresh Clang Release (zero warnings on both), and Clang
ASan+UBSan+leak-detection all pass 17/17 tests (`slice6-gcc-ctest.log`,
`slice6-clang-ctest.log`, `slice6-sanitize-ctest.log`).

### Clean reconstruction

Fresh `git worktree add --detach` at `10fefb345f66b0fdc13e64798a56d3ef83c4d36d`
(which already contains Slices 1-5), Slice 6's diff applied as a patch, the
one new file (`tests/test_phase2f_slice6.cpp`) copied in, diff verified
byte-identical to the working tree, all three build gates, the 335/335 real
MTP binding, the Phase 2D fingerprint, and the real MTP ON/OFF generation
fixture re-run from scratch with identical results.

### Deviations / open questions

- The EOS-inside-verified-prefix synthetic test lands EOS at batch index 0
  rather than the taskblock's illustrative index 1 (i.e. "commit A, then
  commit EOS" rather than "commit A, B=EOS, discard C"): despite substantial
  search effort (see the empirical methodology above), no real-forward
  fixture was found within reasonable time where two *genuinely distinct*
  values are both accepted before a third, EOS-designated value. The
  underlying mechanic under test — a decision batch verified >1 token deep,
  terminating partway through and discarding the unverified-but-already-
  decided remainder — is still exercised faithfully (the queue holds 2
  entries; only the first is ever committed); only the specific illustrative
  shape (non-EOS-first) differs. `test_token_stop_inside_verified_prefix`
  and `test_text_stop_inside_verified_prefix` both *do* achieve the fuller
  "accept, accept, then stop, discard the 3rd" shape using the all-zero
  fixture's reliable `[0,0,0]` full-accept chain, which covers the same
  code path this deviation would have exercised.
- No new natural real partial-reject *search* was run at the session level
  (per the taskblock's explicit "do not burn time searching for one");
  Slice 6's real rejection fixture instead reused the established Slice 3/5
  prompt/position, and happened to additionally exhibit genuine partial-
  accept diagnostics (accepted=1, rejected=1, unused=1) as a bonus, not by
  design.

### Deliberate exclusions (per task instruction)

- No shadow-state promotion into canonical session state, no zero-copy
  speculative cache commit, no bonus-token consumption (Slice 5's bonus
  logits remain diagnostic-only and are never read by Slice 6), no
  stochastic speculative sampling, no adaptive draft depth, no dynamic
  acceptance policy, no acceptance-rate optimization.
- No CUDA/SIMD/CPU-threading/scheduler/HTTP/API/Lumina integration, no
  Tiered Weight Residency, no performance claim anywhere in this slice.

### Verdict

`VERDICT=PHASE2F_SLICE6_GUARDED_GENERATION_INTEGRATION_MATCH`

## Slice 7 — Final correctness closure, real MTP ON/OFF matrix, landing candidate

### Slice 6 EOS closure: a real `[A, EOS, C]` decision batch, `A != EOS`

Slice 6's EOS-inside-a-verified-prefix test used a fixture where EOS landed
at batch index 0 (`[EOS, C]`), not index 1 (`[A, EOS, C]`) as the taskblock's
illustrative example asked for. Investigating why revealed a real
architectural fact, not a testing oversight: every fixture Slice 6 used
(all-zero, one-hot "identity", and the sinusoidally-"textured" eh_proj) is
*memoryless* — each one's MTP and backbone predictions are pure functions of
the current candidate token alone. Chaining a memoryless map can only ever
repeat a fixed point or diverge from it on the very first step; it can never
produce a genuine two-*distinct*-value accepted prefix, because whatever
token step *k* predicts becomes step *k+1*'s candidate, re-entering the same
fixed function.

Closing this required a fixture whose MTP prediction is *not* memoryless —
sensitive to its full input (both the `e_norm`/candidate-token half and the
`h_norm`/hidden-state half of the `eh_proj` concat), so that a step's
prediction can differ from an earlier step's even when they share a
candidate token. Rather than hand-derive such a matrix (a real transformer's
FFN/attention nonlinearities make this intractable to solve analytically by
inspection), this project's established empirical-search methodology was
extended one more time: Phase 2E's own hand-engineered `ModelFixture`
(`tests/test_phase2e.cpp`; vocab 8, embedding 4 — genuinely non-trivial
recurrent+attention dynamics, proven and already in use for Phase 2E's own
tests) was extended with an MTP block, and a **fixed-seed (5150), fully
deterministic pseudo-random search** over `eh_proj` matrices was run against
every 2-token prompt (of 64 possible) whose own natural continuation
produces two distinct values. Exactly one prompt qualified — `[3, 6]`,
naturally continuing `[6, 0, 0, ...]` — and the search found a match on its
107th attempt (of a 20000-attempt budget): an `eh_proj` matrix whose real
MTP draft chain is `[6, 0, 0]`, bit-for-bit matching that prompt's own
natural three-step continuation. The exact matrix is committed verbatim in
`tests/test_phase2f_slice7.cpp`.

With `eos_token_id = 0`, this real chain gives exactly the required shape:
`A = 6`, `EOS = 0`, `C = 0` (the third, never-attempted decision), `A != EOS`.
`test_eos_at_batch_index_1` verifies, against a real (non-mocked) forward
pass: A commits normally; EOS commits and terminates generation
(`finish_reason == eos`); C is never committed (`generated_tokens.size() ==
2`); C produces no canonical callback/event (exactly 2 events, `[6, 0]`);
C leaves no canonical cache residue (final `sequence_length() == 4` —
prompt(2) + committed(2), never 5); final canonical state is bit-identical
to an MTP-OFF lane generating `[6, 0]`; raw/visible text, termination
reason, and diagnostics (`unused_drafts >= 1`) all match.

This last check surfaced a genuine, small gap in Slice 6's own diagnostic
accounting, now fixed: `Qwen35MtpDiagnostics::unused_drafts` previously only
accumulated Slice 5's own verification-time `unused_suffix` (drafts never
individually verified because an earlier one in the same batch was
rejected). It did not account for predictions that *were* verified and
queued but never popped because generation ended first — exactly this
fixture's case (the chain was a genuine full accept; C's "unused" status is
purely a session-level commit-loop fact, invisible to Slice 5's own
accounting). `generate_fresh` now adds `mtp_pending.size()` (whatever is
still queued when the function returns) to `unused_drafts` before
finalizing the result. This is the only change to `qwen35_generation.cpp`'s
existing logic in this slice — three lines, all additive or accounting-only,
no change to any commit/termination/sampling behavior.

### Final Phase 2F architecture

**A. Storage/binding** (Slice 1, 1B): real Qwen3.5-2B-MTP checkpoint
forensics; generic Q8_0 scalar storage decode support; 335/335 real tensor
binding, including the MTP block's `eh_proj`/`enorm`/`hnorm`/
`shared_head_norm` and its dense full-attention block.

**B. NextN numerical execution** (Slice 2): one real MTP/NextN forward
through Oracle's decoded-F32 scalar reference path — `enorm`/`hnorm` →
fusion → `eh_proj` → the appended block's attention+FFN → `shared_head_norm`
→ the tied output head — validated stage-by-stage against a real captured
`graph_mtp` execution.

**C. Proposal chaining** (Slice 4): a bounded, real chain of D0..D(depth-1)
MTP proposals, each depending causally on the previous draft's own output
hidden state (never the target's), with the MTP block's local KV cache
accumulating across the whole chain.

**D. Sequential verification oracle** (Slice 3, 4): one-token-at-a-time
target verification against freshly-forwarded target state, first-rejection
semantics, no rollback needed or used. **This remains the permanent
correctness oracle for all of Phase 2F** — never modified, never weakened,
never routed through any later slice's implementation.

**E. Batched verification / state recovery** (Slice 5): a second,
independent verification strategy — the whole chain evaluated through the
target in one pass, the accepted prefix determined from that pass's own
logits, and any non-canonical suffix discarded via an exact `HybridCache`
boundary snapshot/restore (a single, narrow boundary — not a general
snapshot stack — sized to exactly what Gated DeltaNet recurrent state
requires, source-proven against the pinned reference's own equivalent
fallback). Cross-validated bit-for-bit against D on every synthetic and
real fixture; still fully independent.

**F. Guarded generation integration** (Slice 6, 7): `Qwen35GenerationSession`
gains an off-by-default MTP mode. MTP only ever *predicts*; the real
sampler always independently reconfirms every committed token via the
unmodified Phase 2E per-token commit path, with an internal-consistency
assertion if the two ever disagree (structurally, they cannot, given
deterministic scalar execution and greedy-only MTP mode).

**G. Real MTP ON/OFF parity** (Slice 6, 7): validated on the real 2B MTP
checkpoint across the established capital-of-France and rejection fixtures
plus a broader diverse-prompt, multi-depth matrix (below) — every lane
canonically identical to MTP OFF.

**H. What remains intentionally unoptimized**: everything. See "Phase 2F
scope claim" below for the explicit list.

### Capabilities and boundaries (explicit, per taskblock)

- MTP generation currently preserves canonical target authority: the real,
  freshly-forwarded target's own greedy sample is what gets committed,
  always — MTP predicts, it never decides.
- Reference mode is greedy-only (`sampling.temperature == 0`); stochastic
  MTP configuration fails clearly at preflight rather than attempting
  probabilistic speculative-decoding correction.
- Max draft depth is currently 3 (`{1, 2, 3}` supported, validated at
  preflight; no higher bound, no adaptive depth).
- Shadow verification state is never promoted into canonical session
  state — the shadow `HybridCache` copy is disposable, discarded the moment
  its accounting is read.
- Canonical tokens are always replayed through the ordinary Phase 2E
  single-token commit path — forward, sample-with-assertion, ledger,
  assembler, stop/reasoning bookkeeping, callback, termination check —
  never a second, speculative-specific implementation of any of those
  rules.
- Bonus-token consumption is not implemented: Slice 5's bonus logits remain
  diagnostic-only evidence and are never read by generation.
- No speculative performance claim is made anywhere in Phase 2F. Oracle's
  scalar reference execution is correctness-oriented by design (see Phase
  2C/2D precedent); MTP integration deliberately recomputes real target
  forwards rather than promoting speculative state, and this is by design,
  not an oversight.
- Slice 4's sequential verifier remains a permanent oracle; Slice 5's
  batched verifier remains an independent comparator. Neither is routed
  through the other, in either direction, anywhere in Phase 2F.

### Phase 2F scope claim

If every gate below is green, Phase 2F may claim: **"Validated guarded
Qwen3.5 MTP / NextN speculative generation correctness."**

Phase 2F does **not** claim: optimized speculative decoding; production
speedup; CUDA MTP; adaptive speculation; stochastic speculation; production
batched target execution. No wall-clock timing anywhere in this project's
evidence is used to support a performance claim — every timing figure that
appears in disposable harness output (e.g. "~10s/token" for Oracle's own
scalar CPU path) is incidental logging, not benchmark evidence.

### Real MTP ON/OFF matrix

12 lanes across 8 diverse real prompts on the real 2B MTP checkpoint,
covering every supported depth (1, 2, 3) and every taskblock-suggested
category (simple factual, arithmetic, short reasoning, coding, structured
list, completion-style, a reasoning-marker-producing prompt, and the known
rejection fixture). **Every lane: `PARITY_OK`** — identical canonical token
ids, identical raw/visible text, identical termination, identical final
`HybridCache` fingerprint, MTP OFF vs MTP ON, at every depth:

| prompt | depth | on tokens | finish | fingerprint | opportunities | accepted/rejected/unused | chain mix (full/partial/zero) |
|---|---|---|---|---|---|---|---|
| capital-of-france | 1 | `[248068,271,248069,271,760]` | max_tokens | `0xbbd63c415e657f3e` | 5 | 5/0/0 | 5/0/0 |
| capital-of-france | 2 | (same) | max_tokens | (same) | 3 | 5/0/0 | 3/0/0 |
| capital-of-france | 3 | (same) | max_tokens | (same) | 2 | 5/0/0 | 2/0/0 |
| rejection-fixture | 1 | `[25,198]` | max_tokens | `0x8d12478697ead2be` | 2 | 1/1/0 | 1/0/1 |
| rejection-fixture | 2 | (same) | max_tokens | (same) | 2 | 1/1/1 | 1/0/1 |
| rejection-fixture | 3 | (same) | max_tokens | (same) | 2 | 1/1/1 | 1/0/1 |
| arithmetic ("2 + 2 = ") | 3 | `[19,11,220,18]` | max_tokens | `0xb6137a5efe8d7772` | 2 | 2/2/1 | 0/2/0 |
| short-reasoning | 2 | `[13988,13,198]` | max_tokens | `0xea813727168249a4` | 2 | 2/1/0 | 1/0/1 |
| coding | 1 | `[264,478,292,271]` | max_tokens | `0xc008440c5df2ae3e` | 4 | 3/1/0 | 3/0/1 |
| structured-list | 3 | `[3605,198,17,13]` | max_tokens | `0x687de1b361a4d715` | 2 | 2/2/2 | 0/1/1 |
| simple-factual | 2 | `[10663,321,7062]` | max_tokens | `0x227a2b4870f8fdcd` | 2 | 2/1/0 | 1/1/0 |
| completion-style | 1 | `[11,303,264,22960]` | max_tokens | `0x8665b95d8ff7411e` | 4 | 2/2/0 | 2/0/2 |

Both known-answer fixtures reproduce exactly, at every depth: the
capital-of-France chain (`[248068,271,248069,271,760]`) and its Slice 6
fingerprint (`0xbbd63c415e657f3e`) are bit-for-bit unchanged; the rejection
fixture's canonical beginning (`[25,198]`) is unchanged, with the target=25
rejection now happening organically as part of a full generation-session
run rather than a hand-constructed single-shot query. Real, natural
partial-accept chains occurred organically on 3 of the 8 prompts
(arithmetic: 2; structured-list: 1; simple-factual: 1) — this was not
engineered or searched for, so no "not found" disclaimer is needed. As
required, MTP diagnostics vary meaningfully by depth (more proposed, more
opportunities at shallower depths) while every canonical field stays
identical — proof that depth is purely a *prediction* knob, never a
*decision* knob.

### Landing-review amendment: literal values for the three real/synthetic known-answers

The matrix table above and the EOS closure section earlier both summarize
results; this amendment records the specific literal values a landing
reviewer would want to see spelled out, pulled from evidence already
generated (`slice7-closure/real-mtp-onoff-depth-matrix-run.log`,
`slice7-closure/rejection-evidence-run.log`, and the Slice 7 test source
itself) rather than re-running the full matrix.

**Capital-of-France, depth 3.** MTP ON reproduces the generated token ids
`[248068, 271, 248069, 271, 760]` with finish reason `max_tokens` and final
`HybridCache` fingerprint `0xbbd63c415e657f3e` — bit-identical to MTP OFF
at the same depth and to every other depth (1, 2, 3) in the matrix.

**Rejection fixture, target 25 / draft 279.** A fresh, targeted run
(`slice7_rejection_evidence.cpp`, not the matrix harness) reproduced the
lower-level chain directly: prompt `"The three ingredients you need are"`
(tokens `[760,2250,13565,488,1144,513]`), seed position 5, target argmax
`25`, MTP draft `D0.draft_token = 279` — a confirmed natural mismatch
(`target != draft`), zero-accept (`accepted=0 rejected=1`), lower-level
canonical `[25]`. At the full `Qwen35GenerationSession` level (MTP ON,
depth 1, with an event callback attached), the canonical ledger is
`[25, 198]` and the event stream delivered is also exactly `[25, 198]` —
token `279` appears in neither. The MTP-ON session's final `HybridCache`
fingerprint (`0x8d12478697ead2be`) is bit-identical both to the MTP-OFF
session's fingerprint from the same run and to the matrix table's own
`rejection-fixture` row — an independent cross-check of the same value by
two different harnesses. Because an MTP-disabled/pure-autoregressive run
never proposes or touches `279` by construction, this fingerprint equality
is the proof that no trace of `279` survives into the committed canonical
cache.

**EOS `[6,0,0]` fixture, MTP-ON == MTP-OFF `[6,0]`.** This is a synthetic
fixture (`tests/test_phase2f_slice7.cpp`, `test_eos_at_batch_index_1`),
not a real-checkpoint run; its literal expected values are hard-coded as
`require()` assertions in the committed test source and were verified by
a passing test run (`oracle-phase2f-slice7-tests`, test #18, 18/18) in
GCC/Clang/sanitizer gates on both the working tree and the independent
clean-worktree reconstruction. The literal assertions are: MTP-ON
`generated_tokens == [6, 0]` with `finish_reason == eos`; MTP-OFF and
MTP-ON `generated_tokens`, `generated_text`, and `final_sequence_length`
are all equal; the event stream is exactly `[6, 0]` (never a 3rd event)
with MTP-OFF/ON event token ids and `text_fragment`s equal at every index;
`mtp_diagnostics.unused_drafts >= 1` (the discarded 3rd draft, `C`, is
accounted for); final `HybridCache.sequence_length() == 4` on both lanes
(prompt(2) + committed(2) — `C` never advanced it); and MTP-OFF/ON final
`HybridCache` fingerprints are bit-identical
(`tests/test_phase2f_slice7.cpp:402-447`).

**Diff-stat reconciliation.** Two different insertion counts were cited
for `docs/PHASE_2F.md` at different points while writing this closure
(`814` when the reconstruction worktree was built and verified, `845`
after the reconstruction-results paragraph was appended) purely because
`docs/PHASE_2F.md` is itself one of the diffed files, so each further
closure paragraph — including this reconciliation note — mechanically
grows its own insertion count by construction; the four tracked source/
build files (`CMakeLists.txt`, `include/oracle/runtime/qwen35_generation.hpp`,
`include/oracle/version.hpp`, `src/runtime/qwen35_generation.cpp`) and
the two new test files never moved from their originally-diffed content
and were confirmed byte-identical in the reconstruction worktree. This
is not evidence of drift in the actual engineering change, only of the
documentation describing its own diff while still being written. For
that reason no specific insertion count for `docs/PHASE_2F.md` is quoted
as final here; `git diff --stat 10fefb345f66b0fdc13e64798a56d3ef83c4d36d`
against the frozen pre-commit tree is the authoritative source once no
further doc edits are pending, and is what the landing diff review in
the final report should quote.

### Compiler/sanitizer gates (Slice 7)

Fresh GCC Release, fresh Clang Release (zero warnings on both), and Clang
ASan+UBSan+leak-detection all pass 18/18 tests (`slice7-gcc-ctest.log`,
`slice7-clang-ctest.log`, `slice7-sanitize-ctest.log`).

### Phase 2D / Phase 2E regression (Slice 7)

Canonical Phase 2D fingerprint `0x54825e50fa9398cf` re-verified
bit-identical against the fresh Slice 7 GCC build. `oracle-phase2e-tests`
(the complete existing Phase 2E synthetic suite) passes unchanged; the
expensive real 4B `oracle-phase2e-slice2-fixture-v1` matrix was not
re-run (no source change requires it — `qwen35_generation.cpp`'s only
change this slice is the three-line, purely-additive diagnostics fix
described above).

### Test registration

`oracle-phase2f-slice7-tests` is registered in ordinary CTest
(`CMakeLists.txt`), exactly like every other Phase 2F suite. The real
2B-checkpoint matrix (`slice7_real_matrix.cpp`) remains an explicit,
disposable, manually-invoked harness — CI cannot reasonably bundle a
1.4 GiB checkpoint — but its invocation and complete known-answer evidence
are committed verbatim under `model-reports/.../slice7-closure/`.

### Version landing candidate

`1.1.0-phase2e` → `1.2.0-phase2f`, applied only after every gate above
passed, via the project's single existing version mechanism
(`include/oracle/version.hpp`'s `inline constexpr std::string_view version`,
the only place a version string is defined in the codebase; consumed solely
by `src/runtime/engine.cpp`'s CLI banner — no second version source exists
or was introduced). CMakeLists.txt's own `project(oracle_inference_engine
VERSION 1.0.0 ...)` declaration is the unrelated CMake build-system version
and was intentionally left untouched, consistent with every prior Phase 2F
slice. Post-bump, fresh GCC Release, fresh Clang Release (zero warnings),
and Clang ASan+UBSan+leak-detection builds were re-run and all pass 18/18
tests; the bump was then re-verified bit-identical in the Slice 6+7 clean
detached-worktree reconstruction below.

### Clean reconstruction (Slice 6+7 landing candidate)

Fresh `git worktree add --detach` at `10fefb345f66b0fdc13e64798a56d3ef83c4d36d`
(which already contains Slices 1-5; Slices 6 and 7 were never separately
committed, so this single reconstruction covers both). The combined Slice
6+7 diff (`CMakeLists.txt`, `docs/PHASE_2F.md`,
`include/oracle/runtime/qwen35_generation.hpp`, `include/oracle/version.hpp`,
`src/runtime/qwen35_generation.cpp` — 5 files, 814 insertions/17 deletions)
was applied as a patch, the two untracked test files
(`tests/test_phase2f_slice6.cpp`, `tests/test_phase2f_slice7.cpp`) copied
in, and every one of those seven files verified byte-identical to the
working tree. From that clean worktree: fresh GCC Release, fresh Clang
Release (zero warnings on both), and Clang ASan+UBSan+leak-detection all
pass 18/18 tests; the real 335/335 MTP tensor binding re-verified against
the live checkpoint; and the full 12-lane real MTP ON/OFF depth matrix
(same 8 prompts, same depths) was recompiled against the reconstruction
build and rerun end-to-end, producing output byte-identical to the
original evidence capture (`real-mtp-onoff-depth-matrix-run.log`) —
every finish reason, token sequence, generated text, sequence length,
and `HybridCache` fingerprint matched exactly, and `ALL_PARITY_OK YES`
held across all 12 lanes. No commit was made in the reconstruction
worktree; it was removed after verification.

### Deviations / open questions

None beyond what Slices 1-6 already recorded. The EOS-index-1 closure item
(section above) is the only substantive new engineering in this slice
beyond evidence-gathering, documentation, and the version bump.

### Verdict

`VERDICT=PHASE2F_LANDING_CANDIDATE_MATCH`
