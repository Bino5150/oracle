# Oracle Phase 2E — Real Generation and Generation Safety

**Status: landed, version `1.1.0-phase2e`.** Phase 2E builds from landed
commit `5ecf054f1409fc5cdcf4198cb91a5acfb3ea4d90` (`1.0.0-phase2d`) and adds
real, stateful, safety-guarded autoregressive generation on top of Phase 2D's
already-validated numerical backbone. It changes none of Phase 2D's
numerical execution, quantization decoding, or attention/recurrent math.

## Phase 2E landed capability summary

Built incrementally across Slices 1 through 3C (each documented in full
below, in build order, as the permanent implementation record); this section
is the at-a-glance summary of what is landed as a whole.

- **Generation foundation** (Slice 1): formatted/tokenized prompt ingestion,
  sequential prompt-state advancement through `HybridCache`, deterministic
  greedy autoregressive sampling, accepted-token commit semantics, an exact
  state/token ledger (`final_sequence_length ==
  prompt_tokens.size() + generated_tokens.size()`, enforced on every
  successful return), preflight non-mutation, and runtime-failure state
  reset.
- **Text/event surface** (Slice 3A + correction): one structured
  `Qwen35GenerationEvent` per accepted token via an optional callback, exact
  raw `token_bytes` (always lossless, even when suppressed), UTF-8-safe
  `text_fragment`s assembled by the single authoritative
  `Qwen35IncrementalTextAssembler`, deterministic `U+FFFD` replacement for
  malformed/incomplete sequences, and one authoritative final
  `generated_text` (never independently re-decoded).
- **Termination** (Slice 3B): EOS, configured token-ID stop sequences,
  configured UTF-8 text stop sequences, `max_tokens`, `context_exhausted`,
  a deliberate finish-reason precedence, structured `Qwen35StopMatch`
  metadata, and EOS/stop visible-text suppression — while committed stop
  tokens always remain in the canonical ledger and state (state truth is
  never trimmed for presentation).
- **Streaming holdback** (Slice 3B bounded-holdback correction): a bounded
  rolling ambiguity window replaces whole-generation buffering — safe output
  is released incrementally, mid-generation, the instant it can no longer
  participate in any configured stop; canonical event order and zero
  stop-prefix leakage are preserved exactly.
- **Reasoning safety** (Slice 3C): an explicit, template-agnostic
  reasoning-boundary token contract (never literal strings); prompt-derived
  initial reasoning state; a bounded, accepted-token-only periodic-loop
  detector with a structural false-positive floor; `off`/`stop`/
  `force_close` policies; genuine, atomic reasoning-end token commits
  through the same forward/cache/event path as any sampled token; forced-
  token source metadata; full intervention telemetry; bounded intervention
  escalation; and the `reasoning_loop` finish reason.

## Permanent numerical contracts

Phase 2E continues to use Oracle's permanent scalar reference contract,
unchanged from Phase 2D:

```text
weights:         decoded Q5_K/Q6_K -> F32
activations:     original F32
accumulation:    scalar F32
attention cache: semantic F32
```

TurboQuant Q5_K/Q6_K x Q8_K projection math and its pinned F16 attention cache
remain separately labelled independent-production contracts.

### Phase 2D fingerprint provenance

The canonical standalone Oracle Phase 2D decoded-F32 full-backbone logits
fingerprint, re-verified at every gate throughout Phase 2E, is
`0x54825e50fa9398cf` — for `Qwopus3.5-v3-4B.Q5_K_M.gguf`
(sha256 `a66ea93e1fe0470cd3c405cd858955b59c9126d43088d77dd894e3f309363f04`),
token ID `9419`, position `0`.

**`0x457ba0f32c522743` is not Oracle's standalone decoded-F32 full-backbone
fingerprint.** That value came from the separate diagnostic
external-overrides/TurboQuant bridge path: the independent output-head
closure it performed validated the decoded-F32 *projection stage* using an
externally supplied final activation — it did not establish end-to-end
identity between Oracle's native backbone and the TurboQuant override-fed
backbone. See `model-reports/phase2d-fingerprint-forensic-*/` for the full
forensic root-cause record (not altered by this landing pass).

## Slice 1 API

`Qwen35GenerationSession` owns one bounded `HybridCache` and references the
landed manifest/weight contract. `generate_fresh()` performs a complete fresh
request and leaves the cache at exactly:

```text
prompt token count + accepted generated token count
```

Every accepted generated token is forwarded through
`execute_qwen35_reference_token()` before it is recorded as committed. This
includes EOS. Therefore a successful result never reports a token that is not
represented in recurrent/convolution/attention state.

Preflight validates prompt shape, token range, EOS range, generation bound, and
sampling configuration before the existing session is reset. Runtime failures
reset the state before propagating the exception.

`make_qwen35_chat_generation_request()` reuses the landed Qwen3.5 formatter and
tokenizer with special-token parsing enabled.

## Slice 1 finish reasons

```text
eos
max_tokens
context_exhausted
```

Later Phase 2E slices will add:

```text
stop_sequence
reasoning_loop
cancelled
error
```

## Deliberate limits of Slice 1

This candidate does not yet add:

- text streaming or incremental UTF-8 assembly;
- configured stop sequences;
- reasoning-loop detection, force-close, or escalation;
- cancellation;
- a production generation CLI;
- real Qwopus known-answer token fixtures;
- MTP/NextN;
- CUDA, SIMD, threading, HTTP, or performance claims.

These remain required before Phase 2E can land.

## Slice 2 — real Qwopus deterministic generation fixture

Slice 2 adds Oracle's first real-model, real-tokenizer, real-checkpoint
autoregressive generation known-answer fixture, built entirely on the
Slice 1 `Qwen35GenerationSession` without modifying it. It adds one new tool,
`tools/qwen35_generate.cpp` (`oracle-qwen35-generate`), and no changes to any
landed runtime source.

### Fixed fixture (`oracle-phase2e-slice2-fixture-v1`)

- chat request: system = `"You are a helpful assistant."`, user =
  `"What is the capital of France?"`, `add_generation_prompt = true`;
- formatted via the landed `format_qwen35_chat()` (unchanged);
- tokenized via the landed `Qwen35Tokenizer::encode()` with special-token
  parsing enabled (unchanged) — 28 prompt tokens for the pinned Qwopus
  checkpoint;
- sampling: greedy only, `temperature = 0.0` (deterministic argmax, no RNG
  path is touched — see `Sampler::sample()`);
- `max_generated_tokens = 16`, `maximum_context_tokens = 256`.

Changing any of these values changes the fixture's identity; the fixture id
must be bumped if they ever change.

### What the tool preserves

Per run: the exact formatted prompt text, the complete prompt token-ID
sequence, and for every generated token its token ID, position, and the
sampler's selected-candidate summary (`probability`, `candidate_count` —
trivially `1.0`/`1` under greedy decoding). Per fixture: the decoded text,
finish reason, final sequence length, the model path and (when supplied via
`--model-sha256`) its SHA-256, the Oracle baseline commit/version this slice
was built on (`5ecf054f1409fc5cdcf4198cb91a5acfb3ea4d90`, `1.0.0-phase2d`),
and the numerical contract labels (`decoded-f32-scalar`, `f32-semantic` — no
overrides are used, so this is Oracle's genuine end-to-end reference path,
not the diagnostic external-overrides bridge).

### Determinism

The tool runs the fixture `--repeat` times (default 2), constructing a
**brand-new** `Qwen35GenerationSession` (fresh `HybridCache`) each time, and
compares the generated token-ID sequences first; decoded text is only
compared as a secondary, derived signal (`decode()` is a pure function of
the token IDs). The tool exits non-zero if the token-ID sequences ever
disagree.

### Cost note

Each real forward pass on the pinned Qwopus checkpoint costs roughly 22-23s
of single-threaded scalar CPU time (this is expected and untouched —
Slice 2 does no optimization, threading, or SIMD work). A full default-
parameter fixture run (28 prompt + 16 generated tokens, repeat = 2) is
therefore on the order of 30 minutes wall-clock.

### Deliberate limits of Slice 2

This candidate does not yet add:

- text streaming or incremental UTF-8 assembly;
- configured stop sequences;
- reasoning-loop detection, force-close, or escalation;
- MTP/NextN, speculative decoding, or CUDA;
- any optimization, threading, or SIMD work.

These remain required before Phase 2E can land.

## Slice 3A — token events + UTF-8-safe text surface

Slice 3A makes Slice 1/2's already-validated generation **observable**
through a structured, per-token event and an assembled text surface, without
changing token selection, cache semantics, termination policy, or numerical
behavior. It is strictly observational: `Qwen35GenerationSession` evolved
(constructor now also takes a `const tokenizer::Qwen35Tokenizer&`, and
`generate_fresh()` gained an optional callback parameter), but the sampling
loop, accepted-token ledger, EOS/max-token/context-exhaustion handling, and
preflight/runtime-failure-reset behavior are byte-for-byte the same as
Slice 1 — see the Slice 2 real-fixture regression below for proof.

### The order of operations, unchanged and now observed

```text
1. logits exist from the last committed token;
2. sampler chooses a token;
3. token is validated;
4. token is forwarded through the full model;
5. HybridCache advances by exactly one;
6. token is added to the accepted-token ledger;
7. only then does Slice 3A decode its bytes / assemble text / emit an event.
```

Step 7 is placed textually and structurally after step 6 in
`Qwen35GenerationSession::generate_fresh()`, inside the exact same
`try { ... } catch (...) { state_.reset(); throw; }` block Slice 1 already
uses. There is no separate, weaker-guarded code path for event emission —
if any step 1-6 throws, execution never reaches step 7, and the runtime
failure reset described in the Slice 1 section above applies identically.

### `Qwen35GenerationEvent`

One event per accepted generated token, never for prompt-ingestion tokens:

| field | meaning |
|---|---|
| `token_id`, `position`, `probability`, `candidate_count` | mirror the accepted-token ledger (`Qwen35GeneratedToken`) exactly |
| `generated_index` | 0-based index into `generated_tokens`, i.e. callback emission order |
| `special` | `tokenizer.is_special(token_id)` — control/user-defined type |
| `eos` | `token_id == *manifest.eos_token_id` |
| `token_bytes` | this token's own raw decoded bytes (tokenizer byte-fallback; may be an incomplete UTF-8 fragment on its own — see below) |
| `text_fragment` | the UTF-8-safe slice emitted *now* from the pending-byte buffer; may be empty, may include bytes contributed by earlier tokens |
| `sequence_length` | `HybridCache` length right after this token's commit, always `position + 1` |

`token_bytes` and `text_fragment` are deliberately separate fields — one is
"what this token decoded to on its own," the other is "what became safe to
show the user just now." They are frequently different.

### Callback contract

```cpp
using Qwen35GenerationCallback = std::function<void(const Qwen35GenerationEvent&)>;
Qwen35GenerationResult generate_fresh(const Qwen35GenerationRequest& request,
                                      const Qwen35GenerationCallback& callback = {});
```

- Optional: an empty callback (the default) reproduces Slice 1 exactly.
- Synchronous, in generation order, exactly once per accepted token.
- Cannot alter the sampled token or mutate `HybridCache` — it only ever
  receives a `const&` to an event describing something already committed.
- **Failure policy**: if the callback throws, `generate_fresh()` does not
  catch it separately. The exception propagates through Slice 1's existing
  runtime-failure path, which resets session state to empty before
  rethrowing. The caller never receives a `Qwen35GenerationResult` for a
  generation whose callback failed — there is no partial/"mostly successful"
  outcome. This was a deliberate choice over swallowing callback exceptions:
  Oracle's state-integrity discipline treats a callback failure exactly like
  any other runtime failure, preferring atomicity over exposing a partial
  generation as complete. See `test_callback_failure_resets_session_state`
  in `tests/test_phase2e.cpp`.

### UTF-8 buffering policy (`Qwen35IncrementalTextAssembler`)

Investigated before writing any code: `Qwen35Tokenizer::decode()` decodes
byte-fallback text by accumulating **consecutive normal tokens'** own vocab
strings and calling its private `byte_decode()` once per accumulated run
(flushing on every special token). Individual tokens' stored vocab strings
are always whole, complete GPT2-byte-to-unicode-encoded units — BPE merging
never splits one — so decoding a single token in isolation (via the
existing public `decode()` with a one-token span) always recovers exactly
that token's own raw bytes, correctly and losslessly, even when a multi-byte
character was split across a token boundary by BPE (e.g. `é`, U+00E9,
0xC3 0xA9, tokenized as two separate byte-fallback tokens when no merge rule
combines them). This means **no tokenizer API changes were needed** — Slice
3A calls only the existing public `Qwen35Tokenizer::decode()`, `token_type()`,
and `is_special()`.

`Qwen35IncrementalTextAssembler` (in `qwen35_generation.{hpp,cpp}`, exposed
publicly and unit-tested independently of any session/model) draws a firm
line between two things it hands back per token:

- **`token_bytes`** — that token's own raw decoded bytes, exactly and
  losslessly, straight from `Qwen35Tokenizer::decode()`. Never altered,
  never replaced, never dropped, even when malformed or incomplete on its
  own.
- **`text_fragment`** (and the string `finish()` returns, and therefore
  `Qwen35GenerationResult::generated_text`, their concatenation) — a
  *visible-text* view over the same bytes that is **always valid UTF-8**:
  - a valid multi-byte code point split across a token boundary is held
    pending until it completes, then emitted whole — never early, never
    replaced;
  - a byte sequence that can never be valid UTF-8 (stray continuation byte,
    or a lead byte with malformed continuation bytes) is replaced by the
    Unicode replacement character, `U+FFFD`, one `U+FFFD` per offending
    byte, rather than passed through raw or held forever;
  - **at generation completion**, `Qwen35IncrementalTextAssembler::finish()`
    applies the identical replacement policy to whatever is still pending:
    a trailing sequence that will now never complete is replaced by
    `U+FFFD` rather than flushed raw or silently dropped. `generate_fresh()`
    always calls `finish()` exactly once, regardless of finish reason, and
    appends its result to `generated_text`.

`token_bytes` and `text_fragment` can therefore legitimately disagree for
the same event — that is the point of keeping them separate fields (see
"Canonical token identity and display text must remain separate concepts"
in the task brief this slice was built from). A consumer that needs the
exact raw bytes Oracle decoded for a token (e.g. to reassemble a byte
stream some other way) reads `token_bytes`; a consumer that wants to show
something to a human reads `text_fragment`/`generated_text` and is
guaranteed never to receive invalid UTF-8, including at end of generation.

### Special and unused tokens

`Qwen35IncrementalTextAssembler` does not special-case token types itself —
it always decodes via the tokenizer's own `decode()` (default
`skip_special_tokens = false`) and runs whatever bytes come back through the
same UTF-8 safety net. Concretely:

- a **special** (control/user-defined) token's literal text is included in
  `text_fragment` by default, matching the tokenizer's own contract; the
  `special` event flag is what lets a consumer choose to hide it, rather
  than the assembler forcing it empty;
- an **unused**-type token contributes zero bytes (matching
  `Qwen35Tokenizer::decode()`, which skips `TokenType::unused` entirely) —
  its event still exists (fired in order, with the correct position/ledger
  fields), just with an empty `token_bytes`/`text_fragment`.

### One authoritative decoding path

`Qwen35GenerationResult::generated_text` is produced by the same
`Qwen35IncrementalTextAssembler` instance events are built from inside
`generate_fresh()` — never a second, independent decode. `tools/qwen35_generate.cpp`
was updated to source its reported `decoded_text` from
`result.generated_text` instead of its own separate
`tokenizer.decode(generated_token_ids)` call from Slice 2.

### CLI additions (`oracle-qwen35-generate`)

- `--stream-events` — prints each `Qwen35GenerationEvent` as its own JSON
  line to stdout as it's committed (human/non-`--json` mode only, to avoid
  interleaving with the final summary JSON object; diagnostics stay on
  stderr). Pair with `--json` and `--events-json` for machine consumption
  instead.
- `--events-json <path>` — writes the full per-run event list
  (`{"runs":[[event, event, ...], ...]}`) to a file.
- Raw, possibly non-UTF-8 `token_bytes` are encoded as lowercase hex in all
  event JSON (`token_bytes_hex`); `text_fragment` is always valid UTF-8 by
  construction and is therefore an ordinary escaped JSON string. This keeps
  events and text fragments unambiguously distinguishable, and never emits
  invalid JSON regardless of what a token's byte-fallback content is.

### Known-answer preservation

Slice 3A must not change Slice 2's canonical fixture
(`oracle-phase2e-slice2-fixture-v1`). Re-run through the new event/text path,
at least twice from fresh state, with and without a callback attached —
prompt token IDs, generated token IDs, decoded text, finish reason, and
final sequence length all remained exactly as recorded in Slice 2. See the
evidence directory for this slice for the full comparison.

### Deliberate limits of Slice 3A

This candidate does not yet add:

- arbitrary text stop sequences, multi-token stop matching, or stop trimming;
- reasoning-loop detection, forced `</think>`, or repeated-intervention escalation;
- visible repetition guards;
- cancellation;
- HTTP/SSE/OpenAI-compatible streaming (the callback is designed to support
  a future streaming transport, but none is implemented here);
- MTP/NextN, CUDA, or speculative decoding;
- any optimization, threading, or SIMD work.

EOS/max_tokens/context_exhausted continue to behave exactly as Slice 1
defined them — their policy was not expanded in this slice. These remain
required before Phase 2E can land.

## Slice 3B — termination and stop-sequence semantics

Slice 3B teaches Oracle exactly when generation terminates and what the
returned token ledger, cache state, event stream, and visible text mean
when it does: configured token-ID stops, configured UTF-8 text stops,
explicit stop metadata, a deliberate finish-reason precedence, and EOS
visible-text suppression. It changes none of Slice 1's sampling, cache
layout, or tokenizer semantics, and none of Phase 2D's numerical execution.
(At the time this slice was written it was not yet a complete Phase 2E
landing — see "Deliberate limits of Slice 3B" below for what remained,
all of which is now landed via Slice 3C and this landing pass.)

### The primary rule: model state is authoritative

Once a token successfully passes through `execute_qwen35_reference_token()`
and `HybridCache` advances, it is committed — permanently. A stop condition
discovered *because of* that token never undoes it:

- `Qwen35GenerationResult::generated_tokens` contains every committed
  generated token, including every token that forms a matched stop
  sequence;
- `final_sequence_length` includes those tokens; the Slice 1 invariant
  `final_sequence_length == prompt_tokens.size() + generated_tokens.size()`
  is unchanged and still enforced before every successful return;
- state truth is never trimmed to make presentation prettier.

Presentation is a separate, later concern layered on top: configured stop
material (and EOS) must not appear in `generated_text`, but the tokens that
produced it remain fully committed, fully in the ledger, and still produce
one event each with exact `token_bytes`.

### Request surface

```cpp
struct Qwen35GenerationRequest {
    ...
    std::vector<std::vector<std::uint32_t>> token_stop_sequences;
    std::vector<std::string> text_stop_sequences;
};
```

Both default to empty, which is the exact Slice 1/3A no-stop behavior (see
"No-stop regression" below). Token stops and text stops are deliberately
separate vectors — never collapsed into one ambiguous representation.

### Preflight

Validated in `Qwen35GenerationSession::validate_request()`, before the
session is ever reset, alongside every existing Slice 1 check:

- a token stop sequence must be non-empty, and every token ID in it must be
  inside the model vocabulary;
- a text stop string must be non-empty and valid UTF-8 (checked with the
  same strict validator used nowhere else — no replacement, no fallback: a
  malformed stop string is simply rejected).

**Duplicate policy**: byte-identical text stops or identical token-ID
stop sequences are *not* rejected at preflight. Matching always resolves
ties by earliest `configured_index` (see below), so a duplicate is
redundant, never ambiguous. This is a deliberate choice, not an oversight
— see `test_duplicate_stop_definitions_resolve_by_configured_index`.

### Token stop matching

Checked after every committed generated token by comparing the *exact
trailing window* of `generated_tokens` (ledger token IDs only, length equal
to the configured stop) against each configured token stop. A stop of
length L only becomes checkable once at least L tokens have been generated
— `[..., 10, 20]` can never match a configured `[10, 20, 30]`, regardless
of how suggestive the shared prefix looks; only `[..., 10, 20, 30]` does,
detected the instant `30` commits. If multiple configured token stops
would complete on the same commit, the earliest `configured_index` wins.

### Text stop matching

Checked against Slice 3A's UTF-8-safe visible-text source — the same
`Qwen35IncrementalTextAssembler` fragments events and `generated_text` are
built from — never against raw `token_bytes` directly (a malformed byte
can never be part of a valid-UTF-8 stop string in the first place, since
stop strings are preflight-validated UTF-8; the safe text is what a real
generation session actually searches, see
`test_text_stop_matching_uses_safe_text_not_raw_bytes`). Matching is exact
UTF-8 byte equality: case-sensitive, no Unicode normalization, no regex.
It works identically whether the stop is wholly inside one token, split
across two or several tokens, crosses a UTF-8 multi-byte boundary, or
begins in the middle of one token's decoded fragment with unrelated
content both before and after it in the same token — the accumulated
search text is what matters, not any single token's own fragment. Multiple
configured text stops resolve by earliest byte offset in the visible text,
then by earliest `configured_index` on an exact tie.

### Finish reason and precedence

`Qwen35FinishReason` gains `stop_sequence`, used for both configured token
and configured text stops (they are distinguished through
`Qwen35StopMatch`, not through separate finish-reason enumerators — adding
`token_stop`/`text_stop` as distinct reasons was considered and rejected as
unnecessary given the existing metadata). Evaluated in this explicit order
for the token that was *just* committed:

```text
1. eos
2. configured token stop
3. configured text stop
4. max_tokens
```

...then, only before attempting to sample another token:

```text
5. context_exhausted
```

Consequences, each with a dedicated test:

- EOS wins even if the same token also completes a configured token or
  text stop (`test_finish_precedence_eos_vs_token_stop`,
  `..._eos_vs_text_stop`);
- a token stop wins over a text stop completing on the same commit
  (`..._token_stop_vs_text_stop`);
- a configured stop (either kind) wins over `max_tokens` when both become
  true on the same commit (`..._token_stop_vs_max_tokens`,
  `..._text_stop_vs_max_tokens`);
- `max_tokens` wins over `context_exhausted` when the final allowed
  generated token also exactly fills cache capacity, because the loop
  never returns to the top (where capacity is checked) once `max_tokens`
  breaks it (`..._max_tokens_vs_context_exhausted`);
- plain `context_exhausted` is unchanged from Slice 1
  (`..._plain_context_exhausted`).

This precedence is deliberately encoded, not an accident of `if` ordering,
and Slice 3C's future reasoning-safety precedence is explicitly out of
scope here.

### Stop metadata (`Qwen35StopMatch`)

Present in `Qwen35GenerationResult::stop_match` if and only if
`finish_reason == stop_sequence`; absent (`std::nullopt`) otherwise —
callers never have to infer stop provenance from the finish reason alone.

```cpp
enum class Qwen35StopKind { token_sequence, text_sequence };

struct Qwen35StopMatch {
    Qwen35StopKind kind;
    std::size_t configured_index;
    std::size_t generated_token_begin;   // half-open [begin, end) into
    std::size_t generated_token_end;     // generated_tokens
    std::size_t text_byte_offset;        // where visible-text suppression begins
    std::vector<std::uint32_t> matched_token_ids;
    std::string matched_text;            // text_sequence only
};
```

For a token-sequence match, `[begin, end)` is exactly the matched stop
tokens. For a text-sequence match, it is every generated token that
contributed any byte at or after the match start. `text_byte_offset` is
meaningful for both kinds: the length of the (pre-suppression) visible text
that remains visible before suppression begins.

### Presentation: how suppression actually happens

`Qwen35GenerationSession::generate_fresh()` computes, for whichever finish
condition ended the loop, a single suppression boundary — "entry N is the
last fully-visible one; entry N+1 (if a stop) is visible only up to some
local byte offset; everything after is fully suppressed" — and applies it
uniformly:

- **EOS**: the EOS token's own fragment is always fully suppressed (it is
  always the final entry); this applies with or without any stops
  configured, since EOS never needs a later token to resolve it. No
  buffering is required for this rule specifically.
- **Token stop**: every token in `[generated_token_begin,
  generated_token_end)` is fully suppressed; everything before remains
  fully visible.
- **Text stop**: the token containing the match's first byte is truncated
  to its visible prefix (possibly empty, if the match starts at that
  token's first byte); every token from there onward is fully suppressed
  — including any trailing, unrelated content *after* the match within
  that same token (the `"hello<STOP>garbage-in-same-token"` → `"hello"`
  case, not `"hello<STOP>"` or `"hello<STOP>garbage-in-same-token"`).
- **`max_tokens` / `context_exhausted`**: nothing is suppressed.

`token_bytes` is never touched by any of this — only `text_fragment` (per
event) and `generated_text` (their concatenation, still the one
authoritative visible-text path Slice 3A established).

### Callback/event holdback (bounded rolling holdback)

The callback contract from Slice 3A is unchanged: optional, one event per
committed token, in order, never for an uncommitted token, and a throwing
callback still resets session state via the existing runtime-failure path.
Delivery *timing* is a bounded rolling holdback, not whole-generation
buffering:

- **No stops configured**: delivery is synchronous, immediately after each
  token commits — bit-for-bit the same code path as Slice 3A, with EOS
  suppression layered in locally (EOS is self-announcing; it never needs a
  later token to confirm it, so no holdback applies here even though
  stops aren't configured).
- **Any stop configured**: a just-committed token's event is held only
  while it remains *ambiguous* — while its token-ID and/or visible-text
  contribution could still be part of an in-progress prefix match against
  some configured stop. After each commit, `release_resolved_prefix`
  computes, independently, the longest prefix of the still-pending queue
  that is provably safe under (a) every configured token-ID stop and (b)
  every configured text stop, and releases `min` of the two — the maximal
  prefix safe under *both* windows at once:
  - **Token-ID safety** (`token_safe_prefix_count`): with
    `max_token_stop_length` the length of the longest configured token
    stop, a pending entry at generated-index *i* is safe once
    `current_generated_count - max_token_stop_length >= i` — i.e. once
    enough later tokens have committed that no suffix starting at *i* can
    still extend into a configured token stop. (`max_token_stop_length ==
    0` when no token stops are configured, making every entry immediately
    token-safe.)
  - **Text safety** (`ambiguous_suffix_length` /
    `text_safe_prefix_count`): the longest suffix of the pending
    *visible* text that is a byte-for-byte prefix of any configured text
    stop is the ambiguous tail; every pending entry wholly before that
    tail's start is text-safe. (No configured text stops likewise makes
    every entry immediately text-safe.)
  - Released entries are delivered to the callback in original commit
    order, with `text_fragment` already resolved for that entry (full,
    since a released entry can never itself be inside a match — only the
    *first* still-ambiguous entry ever ends up truncated, and only once
    the match is later confirmed at final resolution).
  - Any bytes/entries still pending when the loop ends (stop match found,
    EOS, `max_tokens`, or `context_exhausted`) are resolved and flushed in
    that same closing pass, exactly as Slice 3A's suppression boundary
    computes it — this is the only place a stop's own matched material
    (and, for a text stop, the truncated boundary token) is suppressed.
  - This reordering is purely a delivery-timing optimization: it never
    changes which tokens are sampled, committed, or ultimately suppressed
    — only *when* an already-decided, already-safe event reaches the
    callback. If the callback throws, the same runtime-failure reset
    applies regardless of whether the throw happened during incremental
    release or the closing resolution pass.

Because release is driven by two independent, conservative safety bounds
(never by "wait for the whole match to resolve"), a long generation with a
short or non-matching configured stop never accumulates the full event
history in memory — pending only ever holds the currently-ambiguous tail.
`test_long_generation_with_short_stop_stays_bounded` and
`test_multi_token_stop_prefix_released_once_no_longer_ambiguous` prove
this by reading `session.state().sequence_length()` from inside the
callback and showing each event's delivery lags its own commit by only a
small, bounded number of later tokens — never by the remaining length of
the generation.

### Known-answer fixtures (real Qwopus checkpoint)

All three re-run at least twice from fresh state; results are byte-for-byte
identical across runs and match what this document records exactly.

**A. No-stop regression** — `oracle-phase2e-slice2-fixture-v1`, unchanged:
generated IDs
`[760, 1156, 369, 9859, 264, 1546, 4145, 3296, 25, 328, 3710, 369, 279, 6511, 314, 9338]`,
text `The user is asking a very simple question: "What is the capital of France`,
`max_tokens`, `final_sequence_length = 44`. Proves stop machinery is
inert when nothing is configured.

**B. Token-stop fixture** — same prompt/greedy config, configured token
stop `[9859, 264]`:

```text
committed generated_tokens: [760, 1156, 369, 9859, 264]
visible text:               "The user is"
finish_reason:               stop_sequence
stop_match.kind:             token_sequence
stop_match.matched_token_ids: [9859, 264]
final_sequence_length:       33  (28 + 5)
```

**C. Text-stop fixture** — same prompt, configured text stop `" user"`:

```text
committed generated_tokens: [760, 1156]
visible text:               "The"
finish_reason:               stop_sequence
stop_match.kind:             text_sequence
stop_match.matched_text:     " user"
final_sequence_length:       30  (28 + 2)
```

Both were captured, not guessed, from the real tokenizer/event path and are
recorded here as part of this slice's versioned evidence.

### CLI additions (`oracle-qwen35-generate`)

- `--stop <text>` (repeatable) — configured text stops.
- `--stop-token-ids <id,id,...>` (repeatable) — one configured token stop
  sequence per occurrence.
- Result JSON gains a `"stop_match"` field per run (`null` when absent).
  `--stream-events`/`--events-json` continue to work unchanged; event JSON
  still encodes raw `token_bytes` as hex and `text_fragment` as an ordinary
  (always-valid) JSON string — ambiguity between the two remains
  impossible by construction.

### Deliberate limits of Slice 3B

This candidate does not yet add:

- `<think>` boundary tracking, reasoning-loop detection, periodic-repeat
  detection, force-close, injected reasoning-end tokens, intervention
  telemetry, or a `reasoning_loop` finish reason (all reserved for
  Slice 3C);
- visible repetition guards outside configured stops;
- cancellation;
- HTTP/SSE/OpenAI-compatible streaming, or Lumina integration;
- MTP/NextN, CUDA, speculative decoding, continuous batching, or any
  optimization/threading/SIMD work.

(At the time this slice was written, Phase 2E was not yet complete; Slice 3C
below is what completed it.)

## Slice 3C — hidden-reasoning loop detection and state-consistent force-close

Slice 3C is the first Phase 2E slice intentionally allowed to *intervene* in
model generation: it can terminate a generation early, or inject tokens the
sampler never chose, when a hidden-reasoning segment has fallen into a
degenerate periodic loop. Everything before it (Slices 1 through 3B) is
purely observational. Because of that, this slice treats **correctness and
state honesty as strictly more important than convenience** — every
intervention still goes through the exact same forward/cache/event
machinery as an ordinary sampled token; nothing is ever faked. This is the
slice that completes the Phase 2E landing (version `1.1.0-phase2e`) — see
"Deliberate limits of Slice 3C" below for what remains explicitly deferred
to later phases.

### Reasoning-boundary contract

```cpp
struct Qwen35ReasoningBoundary {
    std::vector<std::uint32_t> start_tokens;
    std::vector<std::uint32_t> end_tokens;
    bool force_close_supported{false};
};
```

The safety layer never hardcodes a literal string such as `"<think>"` or
`"</think>"` — it consumes only accepted token-ID sequences. A
Qwen3.5-specific caller is responsible for resolving a template's own
markers to exact token IDs via the real tokenizer. `Qwen35GenerationRequest`
carries an optional `reasoning_boundary`; when absent, reasoning-loop
safety can never activate regardless of the configured policy, since there
is nothing to bound "hidden reasoning" by.

**Discovered Qwopus boundary tokens** (real tokenizer, not guessed — see
`model-reports/phase2e-slice3c-*/02-reasoning-boundary-discovery.md` for the
full evidence trail):

| Marker | Token IDs | Text |
|---|---|---|
| `start_tokens` | `[248068, 198]` | `"<think>\n"` |
| `end_tokens` | `[248069, 271]` | `"</think>\n\n"` |

`<think>` = 248068, `</think>` = 248069, `\n` = 198, `\n\n` = 271 (a single
merged BPE token, confirmed via encode/decode round trip). The chosen
sequences mirror exactly what the chat template itself emits around hidden
reasoning (`'<|im_start|>assistant\n<think>\n'` on entry,
`'\n</think>\n\n' + content` on exit) — not just the bare special tokens —
so a force-closed generation lands in the same state the model was actually
trained to see before visible content. This was verified against the same
tokenizer path Slices 1–3B already validate: encoding the canonical Slice 2
chat prompt reproduces the exact same 28-token `prompt_token_ids`,
byte-for-byte, ending in `...,248068,198`.

### Prompt-aware initial reasoning state

The canonical Qwopus formatted prompt already ends inside hidden reasoning
(`...<|im_start|>assistant\n<think>\n`), so generation begins *already*
inside a reasoning segment — this is not an edge case, it is the normal
case. `Qwen35GenerationSession::generate_fresh()` initializes
`reasoning_active` by scanning the tokenized prompt itself: whichever of
`start_tokens`/`end_tokens` occurs *last* as a contiguous subsequence
decides the initial state (neither occurring means reasoning never starts
from the prompt at all). Reasoning can also start entirely from generated
content — the exact same boundary-crossing check runs after every ordinary
commit, in both directions.

### Bounded periodic-loop detector

```cpp
struct Qwen35ReasoningLoopDetection {
    std::size_t period{0};
    std::size_t coverage{0};
};

[[nodiscard]] std::optional<Qwen35ReasoningLoopDetection> qwen35_detect_reasoning_loop(
    const std::vector<std::uint32_t>& trailing_window,
    std::size_t maximum_period,
    std::size_t minimum_repeated_coverage);
```

Deliberately decoupled from `Qwen35GenerationSession`/sampling — exactly
like `Qwen35IncrementalTextAssembler` — so it can be exercised directly
against a synthetic trailing token-ID window in tests. The **input is
always accepted committed token IDs** — never display text, callback
fragments, raw token bytes, whitespace-normalized text, or regex over
visible output — and only ever the tokens generated *since the current
reasoning segment began* (never prompt tokens, never content from a prior,
already-closed segment).

Algorithm: scan only the trailing window handed in (already bounded by the
caller to `inspection_window` and to the current segment's length — the
function itself cannot see anything beyond what it is given, so this is
never an unbounded scan over full generation history) for the *shortest*
period `p` in `[1, maximum_period]` whose repeating pattern covers at least
`minimum_repeated_coverage` trailing tokens **and** at least `3 × p` tokens.
The `3×` rule is a *structural* guarantee, not merely a configured
threshold: a single duplicated token, or even a single duplicated block,
can never qualify — it always takes at least three repeat-units — no matter
how low `minimum_repeated_coverage` is set. Cost is
`O(window_length × maximum_period)`, bounded and constant per commit.

**Configuration** (`Qwen35ReasoningLoopConfig`, all isolated validation
knobs — production defaults intentionally conservative):

| Field | Default | Meaning |
|---|---|---|
| `policy` | `off` | `off` / `stop` / `force_close` |
| `minimum_reasoning_tokens` | 16 | tokens required in-segment before detection activates |
| `inspection_window` | 64 | bounded trailing window ever inspected |
| `maximum_period` | 16 | largest candidate period considered |
| `minimum_repeated_coverage` | 16 | trailing tokens the confirmed period must cover |
| `maximum_interventions` | 3 | force-close attempts allowed before escalating |
| `validation_only_force_trigger_at_reasoning_token_count` | disabled | **never set in production** — see below |

**False-positive protections**: minimum-length gate before any detection
runs; the structural `3×period` + `minimum_repeated_coverage` coverage
floor (no single duplicate pair or block); a hard `maximum_period` bound;
shortest-valid-period preference when multiple periods explain the same
suffix (a genuinely period-2 sequence is trivially also period-4/6/8 — the
detector always reports 2); detection gated strictly on `reasoning_active`
(never runs outside a hidden-reasoning segment, and stops immediately once
one closes — including for identical visible repetition that follows).

**Validation-only trigger**: `validation_only_force_trigger_at_reasoning_token_count`
is a documented escape hatch, disabled by default, that unconditionally
reports a loop once a reasoning segment reaches the given token count,
bypassing the period/coverage gates. It exists solely so the expensive
real-model force-close path (~22.7s/forward on the canonical Qwopus 4B
checkpoint) can be exercised deterministically without waiting for, or
trying to manufacture, genuine pathological repetition from a real
checkpoint. It never alters production detector semantics.

### Policy: off / stop / force_close

```cpp
enum class Qwen35ReasoningLoopPolicy { off, stop, force_close };
```

- **`off`** (default): fully inert. `reasoning_enabled` is
  `false` whenever `policy == off` (or no `reasoning_boundary` is
  configured), and generation takes the *original, unmodified* Slice
  1/3A/3B code path — not a parallel path that merely produces the same
  output. This is what makes "policy off is behaviorally identical to
  pre-Slice-3C Oracle" a structural guarantee rather than a hope; see
  `test_reasoning_policy_off_is_inert` and the unchanged real reasoning-off
  fixtures below.
- **`stop`**: on a confirmed loop, generation terminates immediately with
  `finish_reason == reasoning_loop`. No reasoning-end tokens are injected;
  no further ordinary token is sampled or committed; the ledger/state
  remain exact through the last already-committed token.
- **`force_close`**: on a confirmed loop, Oracle genuinely closes the
  hidden-reasoning segment (below), then resumes ordinary sampling.

### Force-close: genuine, atomic, never cosmetic

On a confirmed loop under `force_close`:

1. ordinary sampling is suspended;
2. the configured `reasoning_boundary.end_tokens` sequence is resolved;
3. **preflight**: the *complete* sequence must fit both remaining context
   capacity and remaining `max_generated_tokens` budget;
4. if it cannot fit, **nothing is committed** — closure fails safely (see
   failure modes below);
5. otherwise every end token commits through the exact same
   `execute_qwen35_reference_token()` / `HybridCache` path an ordinary
   sampled token uses — one real forward pass per forced token, each one
   individually verified (via the same `require_state_advanced` check
   every commit already uses) to advance `HybridCache` by exactly one;
6. each forced token is appended to the canonical `generated_tokens`
   ledger and produces one canonical `Qwen35GenerationEvent` — there is no
   second event path;
7. `reasoning_active` only flips to `false` after the *complete* end
   sequence has committed — never partway;
8. ordinary sampling resumes from the genuinely-advanced state.

**Atomicity** (requirement: a reasoning-end sequence is never half
committed): the inner forced-commit loop runs with no intermediate
token/text-stop check — an in-progress stop-prefix match can never abort
half of a closure. Configured stops resume normal semantics, including the
existing bounded rolling holdback (Slice 3B), immediately after closure
completes; forced-token events participate in that same holdback exactly
like sampled ones.

**Forbidden and structurally impossible by this design**: printing
`</think>` without forwarding it; appending closure text directly to
`generated_text`; emitting a fake event; flipping `reasoning_active`
without model execution; injecting closure tokens into the ledger without
advancing cache; replaying the last sampled token. Every forced token's
`token_bytes` comes from a genuine `Qwen35Tokenizer::decode()` of the
actual forced id, and every forced token's event `sequence_length` is
strictly one greater than the previous event's — proven directly in
`test_force_close_commits_full_end_sequence_and_resumes` by reading
`session.state().sequence_length()` from inside the callback.

### Forced-token source metadata

```cpp
enum class Qwen35GenerationTokenSource { sampled, reasoning_force_close };
```

`Qwen35GenerationEvent::source` defaults to `sampled`, preserving every
existing sampled-event's semantics exactly. For
`source == reasoning_force_close`, `probability`/`candidate_count` are
**never** a fabricated sampling result — they are always exactly `0.0F`/`0`,
a documented sentinel meaning "not applicable" (a forced token was never
sampled). `Qwen35GeneratedToken` (the ledger entry type) is intentionally
left unchanged; a forced token's source is identified via its event and via
the `Qwen35ReasoningIntervention` record that triggered it.

### Reasoning-boundary presentation suppression

A confirmed `reasoning_boundary` start/end sequence is a *template control
signal*, not model-authored content the caller asked to see — the same
distinction Slice 3B already draws for EOS and configured stop matches.
Once a start or end sequence is confirmed committed (genuinely sampled *or*
forced), the **entire configured sequence's** contribution to
`text_fragment`/`generated_text` is suppressed, uniformly, for every token
in it — never `token_bytes` (always exact), never the canonical
`generated_tokens` ledger, `HybridCache`/state, or event ordering, all of
which remain exactly as Slice 3C already established.

**The whole sequence, not just the "control" sub-token, is suppressed.**
For Qwopus, `end_tokens = [248069, 271]` (`"</think>"` + `"\n\n"`) — both
tokens are suppressed, not only `248069`. This was a deliberate choice
between two options:

- suppress only the sub-token(s) the tokenizer marks `special`/`control`
  (for Qwopus this would keep the `\n\n` visible as a paragraph
  separator), or
- suppress the whole configured sequence atomically, exactly like a
  configured stop sequence.

The whole-sequence rule was chosen because `Qwen35ReasoningBoundary` is,
and remains, an opaque token-ID sequence to the generic safety layer — it
never inspects which sub-token within a configured sequence is "the real
marker" versus incidental trailing content, exactly as it never hardcodes
a specific tokenizer's spelling of `<think>`/`</think>` (req: "never
hardcode... a specific tokenizer's spelling"). Introducing a per-sub-token
distinction would require the generic layer to carry template-specific
opinions about sequence structure, which the reasoning-boundary contract
deliberately does not do anywhere else. The tradeoff: a caller that wants
a visible paragraph break after a closed reasoning segment can insert one
itself — `Qwen35ReasoningIntervention`/event `source` already tell it
exactly when a boundary closed, forced or genuine.

**Mechanism**: `PendingEntry` gains a `reasoning_marker_suppressed` flag,
set unconditionally for every forced token (force-close always knows its
own tokens are the marker) and retroactively for genuinely sampled
sequences the instant `generated_ends_with(...)` confirms a full match.
Because a multi-token marker can span more than one commit, the bounded
rolling holdback's `max_token_stop_length` (Slice 3B) is extended to also
cover `max(start_tokens.size(), end_tokens.size())` whenever a boundary is
configured — otherwise the first token of a two-token marker could already
have been incrementally released, unsuppressed, one commit before the
second token confirms the match. Both `release_resolved_prefix()` and the
closing resolution pass check this flag independently of (and in addition
to) the existing EOS/stop suppression before building each entry's visible
fragment. `concat_pending_text()` (used for configured *text* stop
matching) is deliberately untouched — a configured text stop can still
match against a marker's raw bytes if configured to do so; suppression is
a presentation concern only, never a matching concern.

### Max-token and context accounting

Every committed token — sampled or forced — counts toward
`max_generated_tokens`; `generated_tokens.size()` always reflects real
committed work. Forced tokens also consume real context. Both are enforced
as a *preflight* condition before any forced token commits (step 3 above),
so a budget/capacity shortfall can never be discovered mid-closure:

- `boundary.end_tokens.size() > (max_generated_tokens - generated_tokens.size())`
  → `failure_reason = "insufficient_token_budget"`, finish `reasoning_loop`;
- `boundary.end_tokens.size() > (maximum_context_tokens() - state().sequence_length())`
  → `failure_reason = "insufficient_context"`, finish `reasoning_loop`;
- `boundary.end_tokens.empty()` → `failure_reason = "missing_reasoning_end_sequence"`;
- `!boundary.force_close_supported` → `failure_reason = "unsupported_by_template"`;
- intervention budget already exhausted →
  `failure_reason = "intervention_limit_exceeded"` (closure not even
  attempted — see escalation below).

In every failure case, `generated_tokens.size()` is unchanged from the
moment of detection and `reasoning_active` remains `true`.

### Termination precedence

Extends Slice 3B's precedence with reasoning-loop safety inserted between
configured stops and `max_tokens`, for an ordinarily-committed sampled
token:

```
eos  >  configured token stop  >  configured text stop  >  reasoning-loop intervention  >  max_tokens
```

`context_exhausted` is still checked only at the top of the loop, before
attempting another ordinary sample. A token that legitimately completes EOS
or an explicit configured stop is never followed by an unnecessary
reasoning check — the existing `break`-per-condition structure means once
any earlier condition fires, the reasoning check for that commit never even
runs (proven directly by asserting `reasoning_interventions.empty()` in
`test_precedence_eos_vs_reasoning_loop` and the token-/text-stop
equivalents).

A **successful** force-close does *not* itself decide the final
`finish_reason` — it resumes generation, and whatever ordinary condition
happens next wins normally. Example, also the shape of the required real
fixture below: loop detected → force-close succeeds → ordinary sampling
resumes → `max_tokens` is reached → final `finish_reason == max_tokens`
(telemetry still records the earlier intervention).

### Intervention telemetry and escalation

```cpp
struct Qwen35ReasoningIntervention {
    std::size_t generated_index;
    std::size_t reasoning_token_count;
    std::size_t inspected_window_length;
    std::size_t detected_period;
    std::size_t repeated_coverage;
    std::size_t intervention_number;
    Qwen35ReasoningLoopPolicy policy;
    bool closure_attempted;
    bool closure_succeeded;
    std::string failure_reason;
    bool validation_triggered;
};
```

`Qwen35GenerationResult::reasoning_interventions` is the full, in-order
audit trail; `reasoning_active_at_finish` records whether hidden reasoning
was still active the moment generation ended (independent of
`finish_reason` — a `reasoning_loop` finish under `stop` leaves this
`true`; a genuinely closed segment, sampled or forced, leaves it `false`).
`maximum_interventions` bounds how many force-close *attempts* are ever
made; once `intervention_number` would exceed it, the next confirmed loop
is recorded with `closure_attempted = false`,
`failure_reason = "intervention_limit_exceeded"`, and generation finishes
`reasoning_loop` — force-close can never produce an endless intervention
cycle.

### CLI additions (`oracle-qwen35-generate`)

```
--reasoning-loop-policy off|stop|force-close
--reasoning-start-tokens <id,id,...>   --reasoning-end-tokens <id,id,...>
--reasoning-min-tokens <n>             --reasoning-window <n>
--reasoning-max-period <n>             --reasoning-min-coverage <n>
--reasoning-max-interventions <n>
--reasoning-validation-trigger-at <n>  # validation-only, never production
```

A reasoning boundary is only configured (and only then capable of
activating, regardless of policy) when both `--reasoning-start-tokens` and
`--reasoning-end-tokens` are supplied; `force_close_supported` is set
automatically whenever a boundary is configured this way. Result JSON gains
`reasoning_loop_policy`, per-run `reasoning_active_at_finish`,
`reasoning_intervention_count`, and a full `reasoning_interventions` array
(reusing `qwen35_reasoning_intervention_json`); event JSON gains `"source"`
(`"sampled"` or `"reasoning_force_close"`).

### Real Qwopus 4B fixtures

**Reasoning-off regression** (policy `off`, the default — no `--reasoning-*`
flags at all): the existing no-stop, token-stop `[9859,264]`, and text-stop
`" user"` fixtures from Slice 3B were re-run against the Slice 3C binary and
remain byte/token-for-token exact — see
`model-reports/phase2e-slice3c-*/reasoning-off-regression/`.

**Force-close fixture**: the canonical Slice 2 prompt, real Qwopus 4B
checkpoint, genuine prompt ingestion and `HybridCache`, the discovered real
`start_tokens = [248068, 198]` / `end_tokens = [248069, 271]`, and
`--reasoning-validation-trigger-at 2` (the validation-only hook — never a
production flag) to trigger deterministically after a short real sampled
prefix rather than waiting for organic pathological repetition:

```text
idx=0 token_id=760    source=sampled                seq_len=29  text="The"        bytes="The"
idx=1 token_id=1156   source=sampled                seq_len=30  text=" user"      bytes=" user"
idx=2 token_id=248069 source=reasoning_force_close   seq_len=31  text=""          bytes="</think>"  prob=0 cand=0
idx=3 token_id=271    source=reasoning_force_close   seq_len=32  text=""          bytes="\n\n"       prob=0 cand=0
idx=4 token_id=760    source=sampled                seq_len=33  text="The"        bytes="The"
idx=5 token_id=6511   source=sampled                seq_len=34  text=" capital"   bytes=" capital"
idx=6 token_id=314    source=sampled                seq_len=35  text=" of"        bytes=" of"
idx=7 token_id=9338   source=sampled                seq_len=36  text=" France"    bytes=" France"

finish_reason:              max_tokens   (NOT reasoning_loop -- req 20's exact example)
final_sequence_length:      36  (28 prompt + 8 generated)
decoded_text:                "The userThe capital of France"
reasoning_active_at_finish: false
intervention: {generated_index=1, reasoning_token_count=2, detected_period=0,
                repeated_coverage=0, policy=force_close, closure_attempted=true,
                closure_succeeded=true, validation_triggered=true}
```

`sequence_length` advances by exactly 1 per event across the full run,
including both forced tokens — proving one genuine forward pass per forced
token, not a batch or replay. The forced tokens' `probability`/
`candidate_count` are the documented `0`/`0` sentinel, never fabricated.
Both forced events' `token_bytes` remain the exact, lossless decode of
`248069`/`271` (`"</think>"`/`"\n\n"`) — only their `text_fragment` is
suppressed, per "Reasoning-boundary presentation suppression" above.
After closure, ordinary sampling genuinely resumes from the force-closed
state: it produces `"The capital of France"` — a causally coherent
continuation of the injected `</think>\n\n` context (the model is now
answering the question it was asked), not a scripted or replayed value,
and matches the same "The" (760) the model produces from the *original*
prompt in the reasoning-off fixture above, confirming determinism. Full
sampled/forced token IDs, positions, state lengths, and intervention
telemetry are preserved in
`model-reports/phase2e-slice3c-*/force-close-fixture/`.

### Deferred work (Phase 2E landed scope boundary)

Phase 2E, as landed at `1.1.0-phase2e`, deliberately does not add:

- visible-output repetition guards or repetition penalties (out of scope —
  Slice 3C's detector only ever inspects hidden-reasoning content, never
  visible output);
- MTP/NextN, speculative decoding (Phase 2F);
- CUDA, Vulkan, SIMD, multithreaded production kernels, continuous
  batching, scheduler expansion, or prefix caching;
- HTTP/SSE/OpenAI-compatible streaming, or Lumina integration;
- Tiered Weight Residency or SSD/NVMe block streaming.

These remain required before any of that later work begins — this document
describes generation and generation safety only.

## Phase 2E landing

Phase 2E is landed as of version `1.1.0-phase2e`, comprising Slices 1
through 3C plus this landing pass (final documentation, version bump,
full-gate re-verification, and clean reconstruction from the Phase 2D
baseline). See `model-reports/phase2e-clean-landing-*/` for the complete
landing evidence trail.
