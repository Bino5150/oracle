#pragma once

#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/hybrid_cache.hpp"
#include "oracle/runtime/qwen35_chat.hpp"
#include "oracle/runtime/qwen35_forward.hpp"
#include "oracle/runtime/sampler.hpp"
#include "oracle/tokenizer/qwen35_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace oracle::runtime {

enum class Qwen35FinishReason {
    eos,
    stop_sequence,
    max_tokens,
    context_exhausted,
    // Phase 2E Slice 3C: the reasoning-safety layer terminated generation --
    // either a confirmed hidden-reasoning loop under policy `stop`, or a
    // force-close intervention that could not (or, having exhausted its
    // intervention budget, would not) genuinely commit a reasoning-end
    // sequence. See docs/PHASE_2E.md ("Slice 3C").
    reasoning_loop,
};

[[nodiscard]] std::string_view qwen35_finish_reason_name(Qwen35FinishReason reason) noexcept;

// Phase 2E Slice 3B: a configured stop is matched against either the
// committed token-ID ledger (token_sequence) or Slice 3A's UTF-8-safe
// visible-text source (text_sequence) -- never against raw token_bytes.
// See docs/PHASE_2E.md ("Slice 3B") for the full contract.
enum class Qwen35StopKind {
    token_sequence,
    text_sequence,
};

[[nodiscard]] std::string_view qwen35_stop_kind_name(Qwen35StopKind kind) noexcept;

// Phase 2E Slice 3C: distinguishes a token the sampler actually selected
// from one the reasoning-safety layer injected verbatim to force-close a
// hidden reasoning segment. Every event still goes through the one
// canonical event mechanism -- this field is the only thing that tells a
// consumer which kind of commit produced it. See docs/PHASE_2E.md
// ("Slice 3C").
enum class Qwen35GenerationTokenSource {
    sampled,
    reasoning_force_close,
};

[[nodiscard]] std::string_view qwen35_generation_token_source_name(
    Qwen35GenerationTokenSource source) noexcept;

// Phase 2E Slice 3C: a template's hidden-reasoning delimiters, expressed as
// exact accepted-token-ID sequences -- never as literal strings such as
// "<think>"/"</think>". A Qwen3.5-specific layer resolves a template's own
// markers to these token IDs via the real tokenizer (see docs/PHASE_2E.md,
// "Slice 3C", for how Qwopus's own boundaries were discovered); the safety
// layer below only ever consumes token sequences.
struct Qwen35ReasoningBoundary {
    std::vector<std::uint32_t> start_tokens;
    std::vector<std::uint32_t> end_tokens;
    bool force_close_supported{false};
};

// Phase 2E Slice 3C intervention policy once a hidden-reasoning loop is
// confirmed.
enum class Qwen35ReasoningLoopPolicy {
    // Detector is fully inert: existing reasoning-off behavior/fixtures are
    // unchanged bit-for-bit. This is the default.
    off,
    // Terminate generation immediately with finish_reason ==
    // reasoning_loop. No further token -- ordinary or forced -- is sampled
    // or committed.
    stop,
    // Genuinely commit the configured reasoning-end token sequence through
    // the normal full forward/cache path, then resume ordinary sampling.
    // See docs/PHASE_2E.md ("Slice 3C") for the full atomic-closure
    // contract.
    force_close,
};

[[nodiscard]] std::string_view qwen35_reasoning_loop_policy_name(
    Qwen35ReasoningLoopPolicy policy) noexcept;

// Phase 2E Slice 3C: bounded, deterministic periodic-loop-detection
// configuration. Detection only ever consumes accepted committed token IDs
// (never display text, callback fragments, or raw token bytes), only while
// hidden reasoning is active, and only over a bounded trailing window --
// never an unbounded scan over full generation history. See
// docs/PHASE_2E.md ("Slice 3C") for the exact algorithm and defaults.
struct Qwen35ReasoningLoopConfig {
    Qwen35ReasoningLoopPolicy policy{Qwen35ReasoningLoopPolicy::off};

    // No detection at all until at least this many tokens have been
    // committed within the *current* hidden-reasoning segment.
    std::size_t minimum_reasoning_tokens{16};
    // Bounded trailing window (in reasoning-segment tokens) the detector
    // ever inspects; never the full segment history.
    std::size_t inspection_window{64};
    // Largest candidate period the detector will ever consider.
    std::size_t maximum_period{16};
    // A candidate period is only a confirmed loop if it covers at least
    // this many trailing tokens *and* at least 3x its own length --
    // together these are what make a single duplicated token pair
    // insufficient to trigger detection (the 3x rule holds structurally
    // regardless of how this is configured; see
    // qwen35_detect_reasoning_loop below for the exact rule).
    std::size_t minimum_repeated_coverage{16};
    // Force-close interventions allowed before safety gives up on further
    // closure attempts and escalates straight to finishing reasoning_loop.
    std::size_t maximum_interventions{3};

    // Validation-only escape hatch -- never set this in production. When
    // present, and hidden reasoning is active, the detector reports a
    // confirmed loop unconditionally the moment this many reasoning tokens
    // have been committed in the current segment, bypassing the
    // period/coverage gates above. This exists solely so a real-model
    // force-close path can be exercised deterministically and cheaply
    // (without waiting for, or trying to manufacture, genuine pathological
    // repetition from a real checkpoint); it never changes production
    // detector semantics -- default is disabled (nullopt). See
    // docs/PHASE_2E.md ("Slice 3C").
    std::optional<std::size_t> validation_only_force_trigger_at_reasoning_token_count;
};

// Phase 2E Slice 3C: one detection/intervention record, sufficient to audit
// exactly what the safety layer observed and did.
struct Qwen35ReasoningIntervention {
    std::size_t generated_index{0};
    std::size_t reasoning_token_count{0};
    std::size_t inspected_window_length{0};
    std::size_t detected_period{0};
    std::size_t repeated_coverage{0};
    std::size_t intervention_number{0};
    Qwen35ReasoningLoopPolicy policy{Qwen35ReasoningLoopPolicy::off};
    bool closure_attempted{false};
    bool closure_succeeded{false};
    // Empty when closure_succeeded, or under policy stop (closure is never
    // attempted there). Otherwise one of: "insufficient_context",
    // "insufficient_token_budget", "unsupported_by_template",
    // "intervention_limit_exceeded".
    std::string failure_reason;
    bool validation_triggered{false};
};

[[nodiscard]] std::string qwen35_reasoning_intervention_json(
    const Qwen35ReasoningIntervention& intervention);

// Phase 2E Slice 3C: the result of one periodicity check -- the shortest
// confirmed period and how many trailing tokens it covers.
struct Qwen35ReasoningLoopDetection {
    std::size_t period{0};
    std::size_t coverage{0};
};

// Phase 2E Slice 3C bounded periodic-loop detector, deliberately decoupled
// from Qwen35GenerationSession/sampling -- exactly like
// Qwen35IncrementalTextAssembler above -- so it can be exercised directly
// against a synthetic trailing token-ID window. Qwen35GenerationSession
// calls this exact function internally; there is no second implementation.
//
// Scans only `trailing_window` (the caller is responsible for bounding it
// to inspection_window and to the current reasoning segment -- this
// function never sees, and cannot see, anything beyond what it is handed)
// for the *shortest* period p in [1, maximum_period] whose repeating
// pattern covers at least `minimum_repeated_coverage` trailing tokens AND
// at least 3x its own period length. The 3x rule is a structural (not
// merely configured-threshold) guarantee that a single duplicated token --
// or even a single duplicated block -- can never itself qualify, no matter
// how minimum_repeated_coverage is configured: it always takes at least
// three repeat-units. Cost is O(trailing_window.size() * maximum_period),
// never unbounded/quadratic over generation history. Returns nullopt if no
// such period exists.
[[nodiscard]] std::optional<Qwen35ReasoningLoopDetection> qwen35_detect_reasoning_loop(
    const std::vector<std::uint32_t>& trailing_window,
    std::size_t maximum_period,
    std::size_t minimum_repeated_coverage);

// Identifies which configured stop matched, where, and what it matched.
// generated_token_begin/end are a half-open [begin, end) range into
// Qwen35GenerationResult::generated_tokens covering every generated token
// whose visible text is part of (or entirely after) the match -- for a
// token_sequence stop this is exactly the matched stop tokens; for a
// text_sequence stop it is every generated token that contributed any byte
// at or after the match. text_byte_offset is the byte offset, within the
// pre-suppression visible text, where suppression begins (i.e. the visible
// text's length after truncation). matched_token_ids mirrors the token IDs
// in [generated_token_begin, generated_token_end). matched_text is only
// meaningful for text_sequence stops (the exact configured stop string).
struct Qwen35StopMatch {
    Qwen35StopKind kind{Qwen35StopKind::token_sequence};
    std::size_t configured_index{0};
    std::size_t generated_token_begin{0};
    std::size_t generated_token_end{0};
    std::size_t text_byte_offset{0};
    std::vector<std::uint32_t> matched_token_ids;
    std::string matched_text;
};

struct Qwen35GenerationRequest {
    std::vector<std::uint32_t> prompt_tokens;
    std::size_t max_generated_tokens{1};
    SamplingConfig sampling{};

    // Configured stops, checked only against successfully committed
    // generated tokens -- never prompt tokens. Duplicate entries (byte-
    // identical text stops, or identical token-ID sequences) are not
    // rejected; matching resolves ties by earliest configured_index, so a
    // duplicate is simply redundant rather than ambiguous. Empty by
    // default, which preserves Slice 3A behavior exactly (see
    // docs/PHASE_2E.md).
    std::vector<std::vector<std::uint32_t>> token_stop_sequences;
    std::vector<std::string> text_stop_sequences;

    // Phase 2E Slice 3C. Absent (nullopt, the default) means the current
    // template's reasoning boundaries are unconfigured -- reasoning-loop
    // detection can never activate regardless of reasoning_loop.policy in
    // that case, since there is nothing to bound "hidden reasoning" by.
    std::optional<Qwen35ReasoningBoundary> reasoning_boundary;
    Qwen35ReasoningLoopConfig reasoning_loop{};
};

struct Qwen35GeneratedToken {
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    float probability{0.0F};
    std::size_t candidate_count{0};
};

// Phase 2E Slice 3A: a purely observational record of one already-committed
// generated token. An event is only ever constructed after the token has
// been validated, forwarded through the full model, and has advanced
// HybridCache by exactly one -- see Qwen35GenerationSession::generate_fresh.
// token_id/position/probability/candidate_count mirror Qwen35GeneratedToken;
// token_bytes and text_fragment are deliberately separate concepts:
//   token_bytes   -- this token's own raw decoded bytes (tokenizer byte
//                    fallback semantics; may be incomplete UTF-8 on its own
//                    when a multi-byte character was split across tokens).
//                    Always exact and complete, even for EOS or a token
//                    that is itself part of a matched stop sequence.
//   text_fragment -- the UTF-8-safe slice emitted *now* from the session's
//                    pending-byte buffer; may be empty if this token's bytes
//                    only complete a still-incomplete sequence, or may
//                    contain bytes contributed by earlier tokens once a
//                    split sequence completes. Slice 3B additionally
//                    suppresses (empties or truncates) this field -- never
//                    token_bytes -- for EOS and for any token wholly or
//                    partly inside a matched configured stop; the token
//                    itself remains committed and eventfully observed
//                    regardless. See docs/PHASE_2E.md ("Slice 3B").
struct Qwen35GenerationEvent {
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    std::size_t generated_index{0};
    // Phase 2E Slice 3C: for source == reasoning_force_close, probability
    // and candidate_count are NOT a fabricated sampling result -- they are
    // always exactly 0.0F / 0, a documented sentinel meaning "not
    // applicable" (a forced token was never sampled). Check `source` before
    // treating these as a real sampled probability/candidate count.
    float probability{0.0F};
    std::size_t candidate_count{0};
    bool special{false};
    bool eos{false};
    std::string token_bytes;
    std::string text_fragment;
    std::size_t sequence_length{0};
    // Phase 2E Slice 3C. Defaults to `sampled`, preserving every existing
    // sampled-event's semantics exactly.
    Qwen35GenerationTokenSource source{Qwen35GenerationTokenSource::sampled};
};

// Invoked once per accepted token, in generation order -- never for a token
// whose forward/commit did not succeed. Optional: passing no callback (the
// default) preserves Slice 1/3A behavior exactly. A callback must not and
// cannot mutate HybridCache or the sampled token through this interface.
// If a callback throws, generate_fresh() does not treat the generation as
// successful: the exception propagates through the same runtime-failure
// path Slice 1 already uses, which resets session state before rethrowing
// -- including when the throw happens while Slice 3B is releasing
// previously buffered events (see below).
//
// Delivery timing: with no configured stops, delivery is synchronous,
// immediately after each token commits (identical to Slice 3A). With any
// configured stop, Slice 3B cannot yet know whether a just-committed
// token's visible text might need to be suppressed by a stop that only a
// *later* token will confirm, so a committed token's event is held back
// only while it remains ambiguous -- i.e. while its token-ID and/or
// visible-text contribution could still be part of an in-progress prefix
// match against some configured stop. As soon as a pending token can no
// longer participate in any configured stop's completion, its event is
// released immediately (mid-generation, not batched to the end), in
// original commit order, with text_fragment already resolved (full,
// truncated, or empty) -- never token_bytes, which is always exact. Any
// bytes still held back when the generation loop ends (stop found, EOS,
// max_tokens, or context_exhausted) are resolved and flushed in that same
// final pass. This bounded holdback never changes which tokens are sampled
// or committed, and every committed token still ultimately produces
// exactly one ordered event.
// See docs/PHASE_2E.md ("Slice 3B") for the full policy.
using Qwen35GenerationCallback = std::function<void(const Qwen35GenerationEvent&)>;

struct Qwen35GenerationResult {
    Qwen35FinishReason finish_reason{Qwen35FinishReason::max_tokens};
    std::vector<std::uint32_t> prompt_tokens;
    std::vector<Qwen35GeneratedToken> generated_tokens;
    std::size_t final_sequence_length{0};
    // The single authoritative decoding of the generated tokens: the
    // concatenation of every delivered event's text_fragment plus, at
    // completion, any still-pending bytes flushed as-is (see
    // docs/PHASE_2E.md's UTF-8 buffering policy) -- with EOS and matched
    // stop material already suppressed exactly as delivered events are.
    // Never independently re-decoded by another path.
    std::string generated_text;
    // Present if and only if finish_reason == stop_sequence. generated_tokens
    // always contains every committed token, including matched stop tokens
    // -- state truth is never trimmed; this only identifies what was found.
    std::optional<Qwen35StopMatch> stop_match;

    // Phase 2E Slice 3C: whether hidden reasoning was still active at the
    // moment generation finished. Independent of finish_reason -- e.g. a
    // reasoning_loop finish under policy `stop` leaves this true; ordinary
    // completion after a genuinely committed (sampled or force-closed)
    // reasoning-end sequence leaves this false.
    bool reasoning_active_at_finish{false};
    // Every detection/intervention this generation triggered, in
    // chronological order. Empty whenever reasoning_loop.policy == off, the
    // template has no configured reasoning_boundary, or no loop was ever
    // detected.
    std::vector<Qwen35ReasoningIntervention> reasoning_interventions;
};

[[nodiscard]] Qwen35GenerationRequest make_qwen35_chat_generation_request(
    const tokenizer::Qwen35Tokenizer& tokenizer,
    const Qwen35ChatRequest& chat,
    std::size_t max_generated_tokens,
    SamplingConfig sampling = {});

// Phase 2E Slice 3A: incremental, UTF-8-boundary-safe text assembly for a
// sequence of accepted token ids. This is the one authoritative decoding
// path generation events and the final Qwen35GenerationResult::generated_text
// both go through -- there is no second, independent text implementation.
//
// It is intentionally decoupled from Qwen35GenerationSession/sampling so it
// can be exercised directly: construct it from a tokenizer and feed it
// accepted token ids in order.
//
// token_bytes (per append() call) is always the exact, lossless raw bytes
// Qwen35Tokenizer::decode() produces for that one token -- never altered.
//
// text_fragment / the string returned by finish() -- and therefore
// Qwen35GenerationResult::generated_text, which is their concatenation --
// are a *visible-text* view over those same bytes and are always valid
// UTF-8: a valid multi-byte code point split across a token boundary is
// buffered until it completes and then emitted whole (no replacement); a
// byte sequence that can never be valid UTF-8, or one still incomplete when
// finish() is called, is replaced by exactly one U+FFFD per such byte
// rather than passed through raw or silently dropped. See docs/PHASE_2E.md
// for the exact policy and why token_bytes and text_fragment can therefore
// disagree for the same token.
class Qwen35IncrementalTextAssembler {
public:
    explicit Qwen35IncrementalTextAssembler(const tokenizer::Qwen35Tokenizer& tokenizer) noexcept;

    struct AppendResult {
        std::string token_bytes;
        std::string text_fragment;
    };

    // Decodes token_id's own raw bytes (returned verbatim as token_bytes),
    // appends them to the pending buffer, and returns the longest
    // valid-UTF-8 prefix now available to emit as text_fragment -- see the
    // class comment for the exact replacement policy.
    [[nodiscard]] AppendResult append(tokenizer::TokenId token_id);

    // Applies the documented end-of-generation policy to whatever bytes are
    // still pending: a still-incomplete trailing sequence is replaced
    // (never emitted raw), never silently dropped. Safe to call at most
    // once per generation.
    [[nodiscard]] std::string finish();

    [[nodiscard]] const std::string& pending() const noexcept;

private:
    const tokenizer::Qwen35Tokenizer& tokenizer_;
    std::string pending_;
};

class Qwen35GenerationSession {
public:
    Qwen35GenerationSession(const model::Qwen35Manifest& manifest,
                            const model::Qwen35Weights& weights,
                            const tokenizer::Qwen35Tokenizer& tokenizer,
                            std::size_t maximum_context_tokens);

    [[nodiscard]] const HybridCache& state() const noexcept;
    [[nodiscard]] std::size_t maximum_context_tokens() const noexcept;
    void reset() noexcept;

    [[nodiscard]] Qwen35GenerationResult generate_fresh(
        const Qwen35GenerationRequest& request,
        const Qwen35GenerationCallback& callback = {});

private:
    void validate_request(const Qwen35GenerationRequest& request) const;

    const model::Qwen35Manifest& manifest_;
    const model::Qwen35Weights& weights_;
    const tokenizer::Qwen35Tokenizer& tokenizer_;
    HybridCache state_;
};

[[nodiscard]] std::string qwen35_generation_result_text(
    const Qwen35GenerationResult& result);
[[nodiscard]] std::string qwen35_generation_result_json(
    const Qwen35GenerationResult& result);
[[nodiscard]] std::string qwen35_generation_event_json(
    const Qwen35GenerationEvent& event);
[[nodiscard]] std::string qwen35_stop_match_json(const Qwen35StopMatch& stop_match);

}  // namespace oracle::runtime
