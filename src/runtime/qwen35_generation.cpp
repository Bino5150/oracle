#include "oracle/runtime/qwen35_generation.hpp"

#include "oracle/runtime/qwen35_mtp_chain.hpp"
#include "oracle/runtime/qwen35_mtp_chain_verify_batched.hpp"

#include <array>
#include <deque>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace oracle::runtime {
namespace {

[[nodiscard]] std::uint64_t checked_position(std::size_t value) {
    if (value > std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("Qwen3.5 generation position exceeds uint64 range");
    }
    return static_cast<std::uint64_t>(value);
}

void require_state_advanced(const HybridCache& state, std::size_t expected) {
    if (state.sequence_length() != expected) {
        throw std::runtime_error("Qwen3.5 generation hybrid state length mismatch");
    }
}

// The UTF-8 replacement character, U+FFFD, encoded as its 3 UTF-8 bytes.
constexpr std::string_view kReplacementCharacter{"\xEF\xBF\xBD"};

struct Utf8ScanResult {
    // Always valid UTF-8: real code points copied through verbatim, plus
    // one U+FFFD per byte that could not be part of one (see below).
    std::string fragment;
    // How many leading bytes of the input this scan accounted for --
    // either emitted (verbatim or replaced) or, only when `flush_incomplete`
    // is false, left as a still-possibly-completable trailing sequence.
    std::size_t consumed{0};
};

// Scans `bytes` and builds the longest valid-UTF-8 fragment derivable from
// it right now. A trailing lead byte that plausibly begins a longer
// sequence but does not yet have all of its continuation bytes present is
// left unconsumed when `flush_incomplete` is false (the caller retains it
// as pending, to be completed -- or, at generation end, force-flushed --
// later). Byte patterns that can never be valid UTF-8 (stray continuation
// bytes, 0xF8-0xFF, or a lead byte whose continuation bytes are malformed),
// and, when `flush_incomplete` is true, a trailing sequence that will now
// never complete, are each replaced by U+FFFD one byte at a time rather
// than held pending forever or passed through raw -- this is purely a
// *visible-text* policy. It never touches the raw bytes callers separately
// receive as token_bytes, and it does not change Qwen35Tokenizer's own
// decode_utf8_units() byte-fallback behavior, which this deliberately does
// not reuse for visible text.
[[nodiscard]] Utf8ScanResult scan_utf8_prefix(std::string_view bytes, bool flush_incomplete) {
    std::string fragment;
    std::size_t index = 0;
    while (index < bytes.size()) {
        const auto first = static_cast<unsigned char>(bytes[index]);
        std::size_t length = 0;
        std::uint32_t minimum = 0;
        std::uint32_t codepoint = 0;
        if (first < 0x80U) {
            length = 1;
        } else if ((first & 0xE0U) == 0xC0U) {
            length = 2;
            minimum = 0x80U;
            codepoint = first & 0x1FU;
        } else if ((first & 0xF0U) == 0xE0U) {
            length = 3;
            minimum = 0x800U;
            codepoint = first & 0x0FU;
        } else if ((first & 0xF8U) == 0xF0U) {
            length = 4;
            minimum = 0x10000U;
            codepoint = first & 0x07U;
        } else {
            fragment.append(kReplacementCharacter);
            ++index;
            continue;
        }

        if (index + length > bytes.size()) {
            if (!flush_incomplete) {
                break;
            }
            fragment.append(kReplacementCharacter);
            ++index;
            continue;
        }

        bool valid = true;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto continuation = static_cast<unsigned char>(bytes[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) {
                valid = false;
                break;
            }
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        if (valid && (codepoint < minimum || codepoint > 0x10FFFFU ||
                      (codepoint >= 0xD800U && codepoint <= 0xDFFFU))) {
            valid = false;
        }
        if (!valid) {
            fragment.append(kReplacementCharacter);
            ++index;
            continue;
        }
        fragment.append(bytes.substr(index, length));
        index += length;
    }
    return Utf8ScanResult{std::move(fragment), index};
}

// Strict UTF-8 validity check (no replacement, no fallback) used only to
// validate configured text stop strings during preflight -- a malformed
// stop string can never match anything meaningful, so it is rejected
// outright rather than silently normalized.
[[nodiscard]] bool is_valid_utf8(std::string_view text) noexcept {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        std::uint32_t minimum = 0;
        std::uint32_t codepoint = 0;
        if (first < 0x80U) {
            length = 1;
        } else if ((first & 0xE0U) == 0xC0U) {
            length = 2;
            minimum = 0x80U;
            codepoint = first & 0x1FU;
        } else if ((first & 0xF0U) == 0xE0U) {
            length = 3;
            minimum = 0x800U;
            codepoint = first & 0x0FU;
        } else if ((first & 0xF8U) == 0xF0U) {
            length = 4;
            minimum = 0x10000U;
            codepoint = first & 0x07U;
        } else {
            return false;
        }
        if (index + length > text.size()) return false;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto continuation = static_cast<unsigned char>(text[index + offset]);
            if ((continuation & 0xC0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (continuation & 0x3FU);
        }
        if (codepoint < minimum || codepoint > 0x10FFFFU ||
            (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
            return false;
        }
        index += length;
    }
    return true;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 16);
    constexpr char digits[] = "0123456789abcdef";
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character < 0x20U) {
                    output += "\\u00";
                    output.push_back(digits[(character >> 4U) & 0x0FU]);
                    output.push_back(digits[character & 0x0FU]);
                } else {
                    output.push_back(static_cast<char>(character));
                }
        }
    }
    return output;
}

// Deterministic representation for raw (possibly non-UTF-8) token bytes in
// JSON: lowercase hex, two characters per byte. text_fragment is always
// valid UTF-8 by construction and is therefore represented as an ordinary
// escaped JSON string instead.
[[nodiscard]] std::string hex_encode(std::string_view bytes) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(bytes.size() * 2);
    for (const char raw_byte : bytes) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        output.push_back(digits[(byte >> 4U) & 0x0FU]);
        output.push_back(digits[byte & 0x0FU]);
    }
    return output;
}

// Phase 2E Slice 3B: one committed generated token, buffered until the
// generation loop ends and its final (possibly suppressed) visible text can
// be resolved. raw_fragment is Slice 3A's UTF-8-safe fragment for just this
// token, *before* any stop/EOS suppression is applied.
struct PendingEntry {
    std::uint32_t token_id{0};
    std::uint64_t position{0};
    std::size_t generated_index{0};
    float probability{0.0F};
    std::size_t candidate_count{0};
    bool special{false};
    bool eos{false};
    std::size_t sequence_length{0};
    std::string token_bytes;
    std::string raw_fragment;
    // Phase 2E Slice 3C.
    Qwen35GenerationTokenSource source{Qwen35GenerationTokenSource::sampled};
    // Phase 2E Slice 3C correction: true iff this token is part of a
    // confirmed reasoning_boundary start/end sequence -- forced (always) or
    // genuinely sampled (once the full sequence is confirmed committed).
    // Suppresses this entry's contribution to text_fragment/generated_text
    // (never token_bytes, never the ledger) -- see
    // mark_reasoning_boundary_entries() below and docs/PHASE_2E.md
    // ("Reasoning-boundary presentation suppression").
    bool reasoning_marker_suppressed{false};
};

[[nodiscard]] std::string concat_pending_text(const std::vector<PendingEntry>& entries) {
    std::string text;
    for (const PendingEntry& entry : entries) {
        text += entry.raw_fragment;
    }
    return text;
}

struct TokenStopFound {
    std::size_t configured_index{0};
    std::size_t length{0};
};

// Does the tail of `generated_tokens` exactly match any configured token
// stop sequence? Multiple simultaneous matches (necessarily all ending at
// the same, current position) resolve by earliest configured_index.
[[nodiscard]] std::optional<TokenStopFound> find_token_stop_match(
    const std::vector<Qwen35GeneratedToken>& generated_tokens,
    const std::vector<std::vector<std::uint32_t>>& token_stop_sequences) {
    std::optional<TokenStopFound> best;
    for (std::size_t index = 0; index < token_stop_sequences.size(); ++index) {
        const std::vector<std::uint32_t>& stop = token_stop_sequences[index];
        if (stop.size() > generated_tokens.size()) {
            continue;
        }
        const std::size_t start = generated_tokens.size() - stop.size();
        bool matches = true;
        for (std::size_t offset = 0; offset < stop.size(); ++offset) {
            if (generated_tokens[start + offset].token_id != stop[offset]) {
                matches = false;
                break;
            }
        }
        if (!matches) {
            continue;
        }
        if (!best.has_value() || index < best->configured_index) {
            best = TokenStopFound{index, stop.size()};
        }
    }
    return best;
}

struct TextStopFound {
    std::size_t configured_index{0};
    std::size_t offset{0};
};

// Earliest occurrence of any configured text stop within `text`. Ties at
// the same byte offset resolve by earliest configured_index.
[[nodiscard]] std::optional<TextStopFound> find_text_stop_match(
    std::string_view text,
    const std::vector<std::string>& text_stop_sequences) {
    std::optional<TextStopFound> best;
    for (std::size_t index = 0; index < text_stop_sequences.size(); ++index) {
        const std::string& stop = text_stop_sequences[index];
        const std::size_t position = text.find(stop);
        if (position == std::string_view::npos) {
            continue;
        }
        if (!best.has_value() || position < best->offset ||
            (position == best->offset && index < best->configured_index)) {
            best = TextStopFound{index, position};
        }
    }
    return best;
}

// Phase 2E Slice 3B (bounded rolling holdback): how many *leading* pending
// entries can never again be part of a future token-stop match, given the
// current total committed count. A stop of length L can only ever match a
// window ending at some future commit count N' using tokens
// [N'-L+1, N']; an entry at (1-based) commit position P therefore remains
// reachable only while N' <= P + L - 1 for some future N' > current_count,
// i.e. while current_count < P + L - 1. It becomes permanently safe once
// current_count >= P + L - 1, using the *longest* configured length so a
// single boundary is conservative for every configured stop at once.
[[nodiscard]] std::size_t token_safe_prefix_count(const std::vector<PendingEntry>& pending,
                                                   std::size_t current_generated_count,
                                                   std::size_t max_token_stop_length) {
    if (max_token_stop_length == 0) {
        return pending.size();  // no token stops configured: nothing to hold back
    }
    if (current_generated_count < max_token_stop_length) {
        return 0;
    }
    // generated_index is 0-based, so the 1-based commit position is
    // generated_index + 1; safe iff current_generated_count >= position + L - 1
    // iff generated_index <= current_generated_count - max_token_stop_length.
    const std::size_t boundary = current_generated_count - max_token_stop_length;
    std::size_t count = 0;
    while (count < pending.size() && pending[count].generated_index <= boundary) {
        ++count;
    }
    return count;
}

// Longest suffix of `text` that is a (possibly full) prefix of at least one
// configured stop -- the portion that must stay held back because more
// incoming bytes could still complete a match. Returns 0 when nothing in
// `text` could possibly begin any configured stop.
[[nodiscard]] std::size_t ambiguous_suffix_length(std::string_view text,
                                                  const std::vector<std::string>& stops) {
    std::size_t longest = 0;
    for (const std::string& stop : stops) {
        const std::size_t max_check = std::min(text.size(), stop.size());
        for (std::size_t length = max_check; length > longest; --length) {
            if (text.substr(text.size() - length) == std::string_view(stop).substr(0, length)) {
                longest = length;
                break;
            }
        }
    }
    return longest;
}

// How many *leading* pending entries fit entirely within the first
// `safe_byte_boundary` bytes of their concatenated text -- an entry is only
// ever released whole (never split), so a boundary landing mid-entry
// excludes that entire entry.
[[nodiscard]] std::size_t text_safe_prefix_count(const std::vector<PendingEntry>& pending,
                                                 std::size_t safe_byte_boundary) {
    std::size_t cumulative = 0;
    std::size_t count = 0;
    for (; count < pending.size(); ++count) {
        const std::size_t next = cumulative + pending[count].raw_fragment.size();
        if (next > safe_byte_boundary) {
            break;
        }
        cumulative = next;
    }
    return count;
}

// Releases (delivers via `callback`, in order, with full unsuppressed
// text) the maximal leading prefix of `pending` that satisfies BOTH the
// token-stop and text-stop ambiguity windows simultaneously, then removes
// those entries -- the bounded rolling holdback at the heart of Slice 3B.
// Only ever called when the token that was just committed did not itself
// trigger any finish condition, so every released entry's full raw
// fragment is definitionally not stop material.
void release_resolved_prefix(std::vector<PendingEntry>& pending,
                             Qwen35GenerationResult& result,
                             std::size_t max_token_stop_length,
                             const std::vector<std::string>& text_stop_sequences,
                             const Qwen35GenerationCallback& callback) {
    const std::size_t token_safe =
        token_safe_prefix_count(pending, result.generated_tokens.size(), max_token_stop_length);
    const std::string pending_text = concat_pending_text(pending);
    const std::size_t ambiguous = ambiguous_suffix_length(pending_text, text_stop_sequences);
    const std::size_t text_safe_boundary = pending_text.size() - ambiguous;
    const std::size_t text_safe = text_safe_prefix_count(pending, text_safe_boundary);
    const std::size_t release_count = std::min(token_safe, text_safe);

    for (std::size_t i = 0; i < release_count; ++i) {
        const PendingEntry& entry = pending[i];
        // Phase 2E Slice 3C correction: a confirmed reasoning-boundary
        // marker's visible contribution is suppressed here -- never its
        // token_bytes (below, always exact) and never the ledger (already
        // committed before this entry was even queued).
        const std::string visible_fragment =
            entry.reasoning_marker_suppressed ? std::string() : entry.raw_fragment;
        result.generated_text += visible_fragment;
        if (callback) {
            Qwen35GenerationEvent event;
            event.token_id = entry.token_id;
            event.position = entry.position;
            event.generated_index = entry.generated_index;
            event.probability = entry.probability;
            event.candidate_count = entry.candidate_count;
            event.special = entry.special;
            event.eos = entry.eos;
            event.token_bytes = entry.token_bytes;
            event.text_fragment = visible_fragment;
            event.sequence_length = entry.sequence_length;
            event.source = entry.source;
            callback(event);
        }
    }
    pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(release_count));
}

// Phase 2E Slice 3C -----------------------------------------------------

// Does the generated-token ledger end with `needle` as a contiguous
// subsequence? Used to detect a reasoning-start/end boundary crossing on
// each newly committed token.
[[nodiscard]] bool generated_ends_with(const std::vector<Qwen35GeneratedToken>& tokens,
                                       const std::vector<std::uint32_t>& needle) {
    if (needle.empty() || needle.size() > tokens.size()) return false;
    const std::size_t start = tokens.size() - needle.size();
    for (std::size_t i = 0; i < needle.size(); ++i) {
        if (tokens[start + i].token_id != needle[i]) return false;
    }
    return true;
}

// The exclusive end index of the *last* (rightmost) occurrence of `needle`
// as a contiguous subsequence of `ids`, or nullopt if it never occurs.
[[nodiscard]] std::optional<std::size_t> last_occurrence_end(
    const std::vector<std::uint32_t>& ids, const std::vector<std::uint32_t>& needle) {
    if (needle.empty() || needle.size() > ids.size()) return std::nullopt;
    std::optional<std::size_t> found;
    for (std::size_t start = 0; start + needle.size() <= ids.size(); ++start) {
        bool matches = true;
        for (std::size_t i = 0; i < needle.size(); ++i) {
            if (ids[start + i] != needle[i]) {
                matches = false;
                break;
            }
        }
        if (matches) found = start + needle.size();
    }
    return found;
}

// Phase 2E Slice 3C, reasoning-boundary contract section 8: hidden
// reasoning may already be active on entry, because the formatted prompt
// itself can end inside it (Qwopus's own prompt ends in "<think>\n").
// Whichever of start_tokens/end_tokens occurs *last* in the prompt decides
// the initial state; neither occurring means reasoning never starts from
// the prompt at all.
[[nodiscard]] bool initial_reasoning_active(const std::vector<std::uint32_t>& prompt_tokens,
                                            const Qwen35ReasoningBoundary& boundary) {
    const auto start_end = last_occurrence_end(prompt_tokens, boundary.start_tokens);
    const auto end_end = last_occurrence_end(prompt_tokens, boundary.end_tokens);
    if (!start_end.has_value()) return false;
    if (!end_end.has_value()) return true;
    return *start_end > *end_end;
}

// Phase 2E Slice 3C correction: marks the trailing `sequence_length` entries
// of `pending` as reasoning-boundary markers, suppressing their visible
// presentation. Called the instant a start/end sequence is confirmed
// committed -- by construction (bounded holdback is extended to cover
// max(start_tokens.size(), end_tokens.size()), see generate_fresh()) those
// entries are always still present in `pending`, never already released.
// The whole configured sequence is suppressed atomically, exactly like a
// configured stop match (Slice 3B) -- Oracle's safety layer never inspects
// which sub-token within a configured sequence is "the real marker" versus
// incidental trailing content (e.g. a template's own separator whitespace);
// see docs/PHASE_2E.md ("Reasoning-boundary presentation suppression") for
// the full rationale.
void mark_reasoning_boundary_entries(std::vector<PendingEntry>& pending,
                                     std::size_t sequence_length) {
    for (std::size_t i = pending.size() - sequence_length; i < pending.size(); ++i) {
        pending[i].reasoning_marker_suppressed = true;
    }
}

}  // namespace

std::optional<Qwen35ReasoningLoopDetection> qwen35_detect_reasoning_loop(
    const std::vector<std::uint32_t>& trailing_window,
    std::size_t maximum_period,
    std::size_t minimum_repeated_coverage) {
    const std::size_t n = trailing_window.size();
    for (std::size_t period = 1; period <= maximum_period && period < n; ++period) {
        std::size_t matched = 0;
        while (matched < n - period) {
            const std::size_t left = n - 1 - matched;
            if (left < period) break;
            const std::size_t right = left - period;
            if (trailing_window[left] != trailing_window[right]) break;
            ++matched;
        }
        const std::size_t coverage = matched + period;
        if (coverage >= minimum_repeated_coverage && coverage >= 3 * period) {
            return Qwen35ReasoningLoopDetection{period, coverage};
        }
    }
    return std::nullopt;
}

std::string_view qwen35_finish_reason_name(Qwen35FinishReason reason) noexcept {
    switch (reason) {
        case Qwen35FinishReason::eos: return "eos";
        case Qwen35FinishReason::stop_sequence: return "stop_sequence";
        case Qwen35FinishReason::max_tokens: return "max_tokens";
        case Qwen35FinishReason::context_exhausted: return "context_exhausted";
        case Qwen35FinishReason::reasoning_loop: return "reasoning_loop";
    }
    return "unknown";
}

std::string_view qwen35_stop_kind_name(Qwen35StopKind kind) noexcept {
    switch (kind) {
        case Qwen35StopKind::token_sequence: return "token_sequence";
        case Qwen35StopKind::text_sequence: return "text_sequence";
    }
    return "unknown";
}

std::string_view qwen35_generation_token_source_name(
    Qwen35GenerationTokenSource source) noexcept {
    switch (source) {
        case Qwen35GenerationTokenSource::sampled: return "sampled";
        case Qwen35GenerationTokenSource::reasoning_force_close: return "reasoning_force_close";
    }
    return "unknown";
}

std::string_view qwen35_reasoning_loop_policy_name(Qwen35ReasoningLoopPolicy policy) noexcept {
    switch (policy) {
        case Qwen35ReasoningLoopPolicy::off: return "off";
        case Qwen35ReasoningLoopPolicy::stop: return "stop";
        case Qwen35ReasoningLoopPolicy::force_close: return "force_close";
    }
    return "unknown";
}

std::string_view qwen35_mtp_mode_name(Qwen35MtpMode mode) noexcept {
    switch (mode) {
        case Qwen35MtpMode::disabled: return "disabled";
        case Qwen35MtpMode::reference: return "reference";
    }
    return "unknown";
}

std::string qwen35_mtp_diagnostics_json(const Qwen35MtpDiagnostics& diagnostics) {
    std::ostringstream output;
    output << '{' << "\"mtp_enabled\":" << (diagnostics.mtp_enabled ? "true" : "false") << ','
           << "\"requested_draft_depth\":" << diagnostics.requested_draft_depth << ','
           << "\"draft_opportunities\":" << diagnostics.draft_opportunities << ','
           << "\"drafts_proposed\":" << diagnostics.drafts_proposed << ','
           << "\"drafts_accepted\":" << diagnostics.drafts_accepted << ','
           << "\"drafts_rejected\":" << diagnostics.drafts_rejected << ','
           << "\"unused_drafts\":" << diagnostics.unused_drafts << ','
           << "\"verification_count\":" << diagnostics.verification_count << ','
           << "\"full_accept_chains\":" << diagnostics.full_accept_chains << ','
           << "\"partial_accept_chains\":" << diagnostics.partial_accept_chains << ','
           << "\"zero_accept_chains\":" << diagnostics.zero_accept_chains << '}';
    return output.str();
}

Qwen35GenerationRequest make_qwen35_chat_generation_request(
    const tokenizer::Qwen35Tokenizer& tokenizer,
    const Qwen35ChatRequest& chat,
    std::size_t max_generated_tokens,
    SamplingConfig sampling) {
    const std::string formatted = format_qwen35_chat(chat);
    Qwen35GenerationRequest request;
    request.prompt_tokens = tokenizer.encode(
        formatted,
        tokenizer::EncodeOptions{.parse_special_tokens = true});
    request.max_generated_tokens = max_generated_tokens;
    request.sampling = sampling;
    return request;
}

Qwen35IncrementalTextAssembler::Qwen35IncrementalTextAssembler(
    const tokenizer::Qwen35Tokenizer& tokenizer) noexcept
    : tokenizer_(tokenizer) {}

Qwen35IncrementalTextAssembler::AppendResult Qwen35IncrementalTextAssembler::append(
    tokenizer::TokenId token_id) {
    const std::array<tokenizer::TokenId, 1> single{token_id};
    std::string token_bytes = tokenizer_.decode(single, tokenizer::DecodeOptions{});

    pending_ += token_bytes;
    Utf8ScanResult scan = scan_utf8_prefix(pending_, /*flush_incomplete=*/false);
    pending_.erase(0, scan.consumed);

    return AppendResult{std::move(token_bytes), std::move(scan.fragment)};
}

std::string Qwen35IncrementalTextAssembler::finish() {
    // Whatever remains here can only be a still-incomplete trailing
    // sequence -- append() above already consumes anything else on every
    // call -- so this always fully replaces it per the documented policy
    // rather than emitting it raw.
    Utf8ScanResult scan = scan_utf8_prefix(pending_, /*flush_incomplete=*/true);
    pending_.clear();
    return std::move(scan.fragment);
}

const std::string& Qwen35IncrementalTextAssembler::pending() const noexcept { return pending_; }

Qwen35GenerationSession::Qwen35GenerationSession(
    const model::Qwen35Manifest& manifest,
    const model::Qwen35Weights& weights,
    const tokenizer::Qwen35Tokenizer& tokenizer,
    std::size_t maximum_context_tokens)
    : manifest_(manifest),
      weights_(weights),
      tokenizer_(tokenizer),
      state_(manifest, maximum_context_tokens) {}

const HybridCache& Qwen35GenerationSession::state() const noexcept { return state_; }

std::size_t Qwen35GenerationSession::maximum_context_tokens() const noexcept {
    return state_.plan().maximum_tokens;
}

void Qwen35GenerationSession::reset() noexcept { state_.reset(); }

void Qwen35GenerationSession::validate_request(const Qwen35GenerationRequest& request) const {
    if (request.prompt_tokens.empty()) {
        throw std::invalid_argument("Qwen3.5 generation prompt must contain at least one token");
    }
    if (request.max_generated_tokens == 0) {
        throw std::invalid_argument("Qwen3.5 generation max_generated_tokens must be non-zero");
    }
    if (request.prompt_tokens.size() > maximum_context_tokens()) {
        throw std::invalid_argument("Qwen3.5 generation prompt exceeds session context capacity");
    }
    for (const std::uint32_t token_id : request.prompt_tokens) {
        if (token_id >= manifest_.vocabulary_size) {
            throw std::out_of_range("Qwen3.5 generation prompt token exceeds vocabulary");
        }
    }
    if (manifest_.eos_token_id.has_value() &&
        *manifest_.eos_token_id >= manifest_.vocabulary_size) {
        throw std::invalid_argument("Qwen3.5 generation EOS token exceeds vocabulary");
    }

    // Phase 2E Slice 3B: configured stops are validated here too, before
    // the existing session is ever reset, exactly like every other
    // preflight check above.
    for (const std::vector<std::uint32_t>& stop : request.token_stop_sequences) {
        if (stop.empty()) {
            throw std::invalid_argument(
                "Qwen3.5 generation token stop sequence must not be empty");
        }
        for (const std::uint32_t token_id : stop) {
            if (token_id >= manifest_.vocabulary_size) {
                throw std::out_of_range(
                    "Qwen3.5 generation token stop sequence token exceeds vocabulary");
            }
        }
    }
    for (const std::string& stop : request.text_stop_sequences) {
        if (stop.empty()) {
            throw std::invalid_argument(
                "Qwen3.5 generation text stop sequence must not be empty");
        }
        if (!is_valid_utf8(stop)) {
            throw std::invalid_argument(
                "Qwen3.5 generation text stop sequence must be valid UTF-8");
        }
    }

    // Phase 2E Slice 3C: reasoning-boundary token IDs are validated the
    // same way configured stop tokens are. An empty start_tokens/end_tokens
    // sequence is *not* rejected here -- that is a runtime safety outcome
    // (see generate_fresh's "missing_reasoning_end_sequence" handling), not
    // a preflight error.
    if (request.reasoning_boundary.has_value()) {
        for (const std::uint32_t token_id : request.reasoning_boundary->start_tokens) {
            if (token_id >= manifest_.vocabulary_size) {
                throw std::out_of_range(
                    "Qwen3.5 generation reasoning start token exceeds vocabulary");
            }
        }
        for (const std::uint32_t token_id : request.reasoning_boundary->end_tokens) {
            if (token_id >= manifest_.vocabulary_size) {
                throw std::out_of_range(
                    "Qwen3.5 generation reasoning end token exceeds vocabulary");
            }
        }
    }

    // Constructing the sampler is part of preflight so invalid sampling
    // configuration cannot reset or partially mutate an existing session.
    static_cast<void>(Sampler(request.sampling));

    // Phase 2F Slice 6: MTP misuse fails clearly at preflight, exactly like
    // every other configuration error above, rather than silently
    // pretending MTP is active or activating it against an incompatible
    // model.
    if (request.mtp.mode != Qwen35MtpMode::disabled) {
        if (!manifest_.has_mtp()) {
            throw std::invalid_argument(
                "Qwen3.5 generation requested MTP mode but the bound manifest has no MTP block");
        }
        if (request.mtp.max_draft_depth < 1 || request.mtp.max_draft_depth > 3) {
            throw std::invalid_argument(
                "Qwen3.5 generation MTP max_draft_depth must be 1, 2, or 3");
        }
        if (request.sampling.temperature != 0.0F) {
            throw std::invalid_argument(
                "Qwen3.5 generation MTP reference mode requires greedy sampling "
                "(sampling.temperature == 0)");
        }
    }
}

Qwen35GenerationResult Qwen35GenerationSession::generate_fresh(
    const Qwen35GenerationRequest& request,
    const Qwen35GenerationCallback& callback) {
    validate_request(request);
    Sampler sampler(request.sampling);
    const bool stops_configured =
        !request.token_stop_sequences.empty() || !request.text_stop_sequences.empty();
    // Phase 2E Slice 3C: reasoning-loop safety only ever runs with a policy
    // other than `off` AND a configured reasoning_boundary -- without a
    // boundary there is nothing to bound "hidden reasoning" by, so
    // detection can never activate regardless of policy. In either inert
    // case, behavior must stay bit-for-bit identical to pre-Slice-3C
    // Oracle, so generation still takes the original unbuffered fast path.
    const bool reasoning_enabled = request.reasoning_loop.policy != Qwen35ReasoningLoopPolicy::off &&
                                   request.reasoning_boundary.has_value();

    state_.reset();
    try {
        Qwen35ForwardResult current;
        for (std::size_t index = 0; index < request.prompt_tokens.size(); ++index) {
            const std::size_t position = state_.sequence_length();
            current = execute_qwen35_reference_token(
                manifest_,
                weights_,
                request.prompt_tokens[index],
                state_,
                RopePosition::text(checked_position(position)),
                false);
            require_state_advanced(state_, position + 1);
        }

        Qwen35GenerationResult result;
        result.prompt_tokens = request.prompt_tokens;
        result.generated_tokens.reserve(request.max_generated_tokens);

        // Slice 3A: prompt tokens never produce events or contribute to
        // generated_text -- the assembler only ever sees accepted generated
        // tokens, starting from a clean pending-byte buffer.
        Qwen35IncrementalTextAssembler assembler(tokenizer_);

        // Phase 2F Slice 6: the ONE place MTP touches the generation loop.
        // `current` (captured by reference below) always holds, at the
        // point this is called, the forward result of whatever token was
        // most recently committed (or the last prompt token, on the very
        // first call) -- exactly the anchor Slice 2/4/5's MTP contract
        // requires: current.logits is the anchor's own next-token
        // distribution, current.trace.final_norm is its h_nextn, and
        // current.trace.token_id is the anchor token itself.
        //
        // When MTP is enabled and no prediction is already queued, this
        // proposes and shadow-verifies a bounded draft chain (Slices 2-5,
        // completely unmodified) and queues its canonical_tokens as
        // *predictions* -- never as tokens to commit directly. Every call
        // still ends with a real `sampler.sample(current.logits)`, the
        // exact same call Slice 3A/3B/3C's loops already make when MTP is
        // disabled; the only addition is asserting that this real,
        // independently-derived greedy sample agrees with the head of the
        // prediction queue. Because MTP reference mode requires greedy
        // sampling (validate_request) and Oracle's forward is a
        // deterministic scalar function of state, that agreement is
        // guaranteed by construction whenever the shadow state genuinely
        // mirrored canonical state -- a mismatch is therefore treated as an
        // internal-consistency failure, not a recoverable case. This is why
        // MTP cannot change canonical output: the committed token is always
        // the one the ordinary sampler independently produces from a real,
        // freshly (re)computed forward on the real canonical HybridCache
        // (see docs/PHASE_2F.md, "Slice 6", "target state commit model").
        std::deque<std::uint32_t> mtp_pending;
        Qwen35MtpDiagnostics mtp_diag;
        mtp_diag.mtp_enabled = request.mtp.mode != Qwen35MtpMode::disabled;
        mtp_diag.requested_draft_depth = request.mtp.max_draft_depth;

        auto obtain_next_token = [&]() -> SampleResult {
            if (request.mtp.mode != Qwen35MtpMode::disabled && mtp_pending.empty()) {
                // Draft depth bounding (taskblock section 8): never propose
                // speculative tokens that a known hard boundary already
                // forbids. Both quantities are guaranteed >= 1 here -- the
                // loop's own top-of-iteration context_exhausted check, and
                // the previous iteration's own max_tokens check, already
                // guard against either being zero by the time a new token
                // is about to be sampled.
                const std::size_t remaining_budget =
                    request.max_generated_tokens - result.generated_tokens.size();
                const std::size_t remaining_capacity =
                    maximum_context_tokens() - state_.sequence_length();
                std::size_t effective_depth = request.mtp.max_draft_depth;
                effective_depth = std::min(effective_depth, remaining_budget);
                effective_depth = std::min(effective_depth, remaining_capacity);

                if (effective_depth > 0) {
                    const Qwen35MtpDraftChain chain = generate_qwen35_mtp_draft_chain(
                        manifest_, weights_, current.trace.token_id, current.trace.final_norm,
                        state_.sequence_length() - 1,
                        static_cast<std::uint32_t>(effective_depth), false);

                    // Shadow verification: a disposable copy of canonical
                    // state, discarded the moment this block ends. The live
                    // session cache (state_) is never touched here -- see
                    // taskblock section 9.
                    HybridCache shadow_state = state_;
                    const Qwen35MtpChainAccountingBatched accounting =
                        verify_qwen35_mtp_draft_chain_batched(
                            manifest_, weights_, chain, current.logits, shadow_state,
                            static_cast<std::uint32_t>(effective_depth));

                    ++mtp_diag.draft_opportunities;
                    mtp_diag.drafts_proposed += accounting.proposed;
                    mtp_diag.drafts_accepted += accounting.accepted;
                    mtp_diag.drafts_rejected += accounting.rejected;
                    mtp_diag.unused_drafts += accounting.unused_suffix;
                    mtp_diag.verification_count += accounting.verification_count;
                    if (accounting.full_chain_accepted) {
                        ++mtp_diag.full_accept_chains;
                    } else if (accounting.accepted == 0) {
                        ++mtp_diag.zero_accept_chains;
                    } else {
                        ++mtp_diag.partial_accept_chains;
                    }

                    for (const std::uint32_t token_id : accounting.canonical_tokens) {
                        mtp_pending.push_back(token_id);
                    }
                }
            }

            // The one, unmodified sampling call every committed token in
            // this function has always gone through. No skipping, no
            // MTP-specific substitute value.
            const SampleResult sampled = sampler.sample(current.logits);
            if (!mtp_pending.empty()) {
                const std::uint32_t predicted = mtp_pending.front();
                mtp_pending.pop_front();
                if (sampled.token_id != predicted) {
                    throw std::runtime_error(
                        "Qwen3.5 MTP shadow-verified prediction disagreed with the real "
                        "canonical greedy sample -- internal consistency failure");
                }
            }
            return sampled;
        };

        if (!stops_configured && !reasoning_enabled) {
            // No stops configured and reasoning safety inert: identical timing/ordering to Slice 3A
            // (synchronous per-token delivery), with Slice 3B's EOS
            // visible-text suppression layered in. EOS never needs a later
            // token to confirm it, so no buffering is required for this
            // one rule -- see docs/PHASE_2E.md ("Slice 3B").
            for (;;) {
                if (state_.sequence_length() >= maximum_context_tokens()) {
                    result.finish_reason = Qwen35FinishReason::context_exhausted;
                    break;
                }

                const SampleResult sampled = obtain_next_token();
                if (sampled.token_id >= manifest_.vocabulary_size) {
                    throw std::runtime_error(
                        "Qwen3.5 generation sampler returned an invalid token id");
                }

                const std::size_t accepted_position = state_.sequence_length();
                current = execute_qwen35_reference_token(
                    manifest_,
                    weights_,
                    sampled.token_id,
                    state_,
                    RopePosition::text(checked_position(accepted_position)),
                    false);
                require_state_advanced(state_, accepted_position + 1);

                result.generated_tokens.push_back({
                    sampled.token_id,
                    checked_position(accepted_position),
                    sampled.probability,
                    sampled.candidate_count,
                });

                const bool is_eos = manifest_.eos_token_id.has_value() &&
                                    sampled.token_id == *manifest_.eos_token_id;
                const Qwen35IncrementalTextAssembler::AppendResult decoded =
                    assembler.append(sampled.token_id);
                const std::string visible_fragment =
                    is_eos ? std::string() : decoded.text_fragment;
                result.generated_text += visible_fragment;

                if (callback) {
                    Qwen35GenerationEvent event;
                    event.token_id = sampled.token_id;
                    event.position = checked_position(accepted_position);
                    event.generated_index = result.generated_tokens.size() - 1;
                    event.probability = sampled.probability;
                    event.candidate_count = sampled.candidate_count;
                    event.special = tokenizer_.is_special(sampled.token_id);
                    event.eos = is_eos;
                    event.token_bytes = decoded.token_bytes;
                    event.text_fragment = visible_fragment;
                    event.sequence_length = state_.sequence_length();
                    callback(event);
                }

                if (is_eos) {
                    result.finish_reason = Qwen35FinishReason::eos;
                    break;
                }
                if (result.generated_tokens.size() >= request.max_generated_tokens) {
                    result.finish_reason = Qwen35FinishReason::max_tokens;
                    break;
                }
            }

            const std::string trailing_flush = assembler.finish();
            if (result.finish_reason != Qwen35FinishReason::eos) {
                result.generated_text += trailing_flush;
            }
        } else {
            // Stops are configured: a match can only be confirmed once
            // enough later tokens (or bytes) have arrived, so an event is
            // held only while it remains ambiguous -- as soon as it can no
            // longer participate in any configured stop, it is released
            // (delivered via callback, in order) immediately, bounding how
            // much this loop ever holds onto at once. See
            // docs/PHASE_2E.md ("Slice 3B") for the bounded rolling
            // holdback contract.
            std::size_t max_token_stop_length = 0;
            for (const std::vector<std::uint32_t>& stop : request.token_stop_sequences) {
                max_token_stop_length = std::max(max_token_stop_length, stop.size());
            }
            // Phase 2E Slice 3C correction: a genuinely (not forced) sampled
            // reasoning-boundary sequence can span multiple commits exactly
            // like a configured token stop, so the same bounded holdback
            // must also hold an entry back until it is confirmed safe from
            // completing (or being part of) a start/end marker -- otherwise
            // an earlier token in the sequence could already have been
            // released, unsuppressed, before the match is even confirmed.
            if (reasoning_enabled) {
                max_token_stop_length =
                    std::max(max_token_stop_length,
                            std::max(request.reasoning_boundary->start_tokens.size(),
                                    request.reasoning_boundary->end_tokens.size()));
            }

            std::vector<PendingEntry> pending;
            bool stop_matched = false;
            Qwen35StopKind stop_kind{Qwen35StopKind::token_sequence};
            std::size_t stop_configured_index = 0;
            std::size_t stop_token_match_length = 0;
            std::size_t stop_detected_text_offset = 0;
            std::string stop_matched_text;

            // Phase 2E Slice 3C reasoning-safety state. reasoning_active
            // must account for the prompt itself (Qwopus's own prompt ends
            // inside "<think>\n") -- see initial_reasoning_active().
            bool reasoning_active = reasoning_enabled &&
                initial_reasoning_active(request.prompt_tokens, *request.reasoning_boundary);
            std::size_t reasoning_segment_start_index = 0;
            std::size_t reasoning_intervention_count = 0;

            for (;;) {
                if (state_.sequence_length() >= maximum_context_tokens()) {
                    result.finish_reason = Qwen35FinishReason::context_exhausted;
                    break;
                }

                const SampleResult sampled = obtain_next_token();
                if (sampled.token_id >= manifest_.vocabulary_size) {
                    throw std::runtime_error(
                        "Qwen3.5 generation sampler returned an invalid token id");
                }

                const std::size_t accepted_position = state_.sequence_length();
                current = execute_qwen35_reference_token(
                    manifest_,
                    weights_,
                    sampled.token_id,
                    state_,
                    RopePosition::text(checked_position(accepted_position)),
                    false);
                require_state_advanced(state_, accepted_position + 1);

                result.generated_tokens.push_back({
                    sampled.token_id,
                    checked_position(accepted_position),
                    sampled.probability,
                    sampled.candidate_count,
                });

                const bool is_eos = manifest_.eos_token_id.has_value() &&
                                    sampled.token_id == *manifest_.eos_token_id;
                const Qwen35IncrementalTextAssembler::AppendResult decoded =
                    assembler.append(sampled.token_id);

                pending.push_back(PendingEntry{
                    sampled.token_id,
                    checked_position(accepted_position),
                    result.generated_tokens.size() - 1,
                    sampled.probability,
                    sampled.candidate_count,
                    tokenizer_.is_special(sampled.token_id),
                    is_eos,
                    state_.sequence_length(),
                    decoded.token_bytes,
                    decoded.text_fragment,
                });

                // Termination precedence: eos > configured token stop >
                // configured text stop > max_tokens > (next-iteration)
                // context_exhausted. Each condition is evaluated only for
                // the token that was just committed; an earlier stop would
                // already have ended generation on an earlier iteration.
                if (is_eos) {
                    result.finish_reason = Qwen35FinishReason::eos;
                    break;
                }
                if (const auto token_match = find_token_stop_match(
                        result.generated_tokens, request.token_stop_sequences)) {
                    stop_matched = true;
                    stop_kind = Qwen35StopKind::token_sequence;
                    stop_configured_index = token_match->configured_index;
                    stop_token_match_length = token_match->length;
                    result.finish_reason = Qwen35FinishReason::stop_sequence;
                    break;
                }
                const std::string pending_text_so_far = concat_pending_text(pending);
                if (const auto text_match =
                        find_text_stop_match(pending_text_so_far, request.text_stop_sequences)) {
                    stop_matched = true;
                    stop_kind = Qwen35StopKind::text_sequence;
                    stop_configured_index = text_match->configured_index;
                    stop_detected_text_offset = text_match->offset;
                    stop_matched_text = request.text_stop_sequences[text_match->configured_index];
                    result.finish_reason = Qwen35FinishReason::stop_sequence;
                    break;
                }

                // Phase 2E Slice 3C, precedence position 4 (after eos/token
                // stop/text stop, before max_tokens): track the hidden-
                // reasoning boundary crossing on this commit, then run
                // bounded periodic-loop detection only while reasoning is
                // active. A successful force-close does not terminate this
                // iteration -- it falls through to the max_tokens check
                // below exactly like an ordinary commit would, since forced
                // tokens count toward the same budget (req 18).
                bool reasoning_terminal = false;
                if (reasoning_enabled) {
                    if (reasoning_active &&
                        generated_ends_with(result.generated_tokens,
                                            request.reasoning_boundary->end_tokens)) {
                        reasoning_active = false;
                        // Genuinely sampled (not forced) end marker: the
                        // extended holdback above guarantees every token of
                        // this sequence is still in `pending` right now.
                        mark_reasoning_boundary_entries(
                            pending, request.reasoning_boundary->end_tokens.size());
                    } else if (!reasoning_active &&
                              generated_ends_with(result.generated_tokens,
                                                  request.reasoning_boundary->start_tokens)) {
                        reasoning_active = true;
                        reasoning_segment_start_index = result.generated_tokens.size();
                        mark_reasoning_boundary_entries(
                            pending, request.reasoning_boundary->start_tokens.size());
                    }

                    if (reasoning_active) {
                        const std::size_t reasoning_token_count =
                            result.generated_tokens.size() - reasoning_segment_start_index;
                        const Qwen35ReasoningLoopConfig& loop_config = request.reasoning_loop;

                        bool triggered = false;
                        bool validation_triggered = false;
                        Qwen35ReasoningLoopDetection detection{};
                        std::size_t window_length = 0;

                        if (reasoning_token_count >= loop_config.minimum_reasoning_tokens) {
                            window_length =
                                std::min(loop_config.inspection_window, reasoning_token_count);
                            std::vector<std::uint32_t> window_ids;
                            window_ids.reserve(window_length);
                            const std::size_t window_start =
                                result.generated_tokens.size() - window_length;
                            for (std::size_t i = window_start; i < result.generated_tokens.size();
                                 ++i) {
                                window_ids.push_back(result.generated_tokens[i].token_id);
                            }
                            if (const auto found = qwen35_detect_reasoning_loop(
                                    window_ids, loop_config.maximum_period,
                                    loop_config.minimum_repeated_coverage)) {
                                detection = *found;
                                triggered = true;
                            }
                        }

                        if (!triggered &&
                            loop_config.validation_only_force_trigger_at_reasoning_token_count
                                .has_value() &&
                            reasoning_token_count >=
                                *loop_config
                                     .validation_only_force_trigger_at_reasoning_token_count) {
                            triggered = true;
                            validation_triggered = true;
                            if (window_length == 0) {
                                window_length = std::min(loop_config.inspection_window,
                                                         reasoning_token_count);
                            }
                        }

                        if (triggered) {
                            Qwen35ReasoningIntervention record;
                            record.generated_index = result.generated_tokens.size() - 1;
                            record.reasoning_token_count = reasoning_token_count;
                            record.inspected_window_length = window_length;
                            record.detected_period = detection.period;
                            record.repeated_coverage = detection.coverage;
                            record.policy = loop_config.policy;
                            record.validation_triggered = validation_triggered;
                            ++reasoning_intervention_count;
                            record.intervention_number = reasoning_intervention_count;

                            if (loop_config.policy == Qwen35ReasoningLoopPolicy::stop) {
                                record.closure_attempted = false;
                                record.closure_succeeded = false;
                                result.reasoning_interventions.push_back(record);
                                result.finish_reason = Qwen35FinishReason::reasoning_loop;
                                reasoning_terminal = true;
                            } else if (reasoning_intervention_count >
                                      loop_config.maximum_interventions) {
                                record.closure_attempted = false;
                                record.closure_succeeded = false;
                                record.failure_reason = "intervention_limit_exceeded";
                                result.reasoning_interventions.push_back(record);
                                result.finish_reason = Qwen35FinishReason::reasoning_loop;
                                reasoning_terminal = true;
                            } else {
                                const Qwen35ReasoningBoundary& boundary =
                                    *request.reasoning_boundary;
                                std::string failure;
                                if (boundary.end_tokens.empty()) {
                                    failure = "missing_reasoning_end_sequence";
                                } else if (!boundary.force_close_supported) {
                                    failure = "unsupported_by_template";
                                } else {
                                    const std::size_t remaining_context =
                                        maximum_context_tokens() - state_.sequence_length();
                                    const std::size_t remaining_budget =
                                        request.max_generated_tokens -
                                        result.generated_tokens.size();
                                    if (boundary.end_tokens.size() > remaining_context) {
                                        failure = "insufficient_context";
                                    } else if (boundary.end_tokens.size() > remaining_budget) {
                                        failure = "insufficient_token_budget";
                                    }
                                }

                                if (!failure.empty()) {
                                    record.closure_attempted = true;
                                    record.closure_succeeded = false;
                                    record.failure_reason = failure;
                                    result.reasoning_interventions.push_back(record);
                                    result.finish_reason = Qwen35FinishReason::reasoning_loop;
                                    reasoning_terminal = true;
                                } else {
                                    // Atomic force-close: every reasoning-end
                                    // token commits through the normal full
                                    // forward/cache path before
                                    // reasoning_active changes or any of
                                    // these entries are released -- no
                                    // intermediate stop-prefix check runs
                                    // during this inner loop (req 24).
                                    for (const std::uint32_t forced_id : boundary.end_tokens) {
                                        const std::size_t forced_position =
                                            state_.sequence_length();
                                        current = execute_qwen35_reference_token(
                                            manifest_, weights_, forced_id, state_,
                                            RopePosition::text(checked_position(forced_position)),
                                            false);
                                        require_state_advanced(state_, forced_position + 1);

                                        result.generated_tokens.push_back({
                                            forced_id,
                                            checked_position(forced_position),
                                            0.0F,
                                            0,
                                        });
                                        const Qwen35IncrementalTextAssembler::AppendResult
                                            forced_decoded = assembler.append(forced_id);
                                        pending.push_back(PendingEntry{
                                            forced_id,
                                            checked_position(forced_position),
                                            result.generated_tokens.size() - 1,
                                            0.0F,
                                            0,
                                            tokenizer_.is_special(forced_id),
                                            false,
                                            state_.sequence_length(),
                                            forced_decoded.token_bytes,
                                            forced_decoded.text_fragment,
                                            Qwen35GenerationTokenSource::reasoning_force_close,
                                            // Every forced token is, by definition, part of the
                                            // reasoning-end sequence -- always suppressed from
                                            // visible presentation (Slice 3C correction).
                                            true,
                                        });
                                    }
                                    reasoning_active = false;
                                    record.closure_attempted = true;
                                    record.closure_succeeded = true;
                                    result.reasoning_interventions.push_back(record);
                                    release_resolved_prefix(pending, result,
                                                            max_token_stop_length,
                                                            request.text_stop_sequences,
                                                            callback);
                                }
                            }
                        }
                    }
                }
                if (reasoning_terminal) {
                    break;
                }

                if (result.generated_tokens.size() >= request.max_generated_tokens) {
                    result.finish_reason = Qwen35FinishReason::max_tokens;
                    break;
                }

                // No finish condition was triggered by this commit: release
                // whatever leading prefix of `pending` can no longer
                // participate in any future stop match, with its full,
                // unsuppressed text -- this is the incremental half of the
                // bounded rolling holdback.
                release_resolved_prefix(pending, result, max_token_stop_length,
                                        request.text_stop_sequences, callback);
            }

            // Phase 2E Slice 3C: whatever reasoning_active ended up as when
            // the loop above exited -- true only for a reasoning_loop
            // finish under policy `stop` (or context_exhausted/max_tokens
            // reached while still inside hidden reasoning); false whenever
            // reasoning genuinely closed (sampled or force-closed).
            result.reasoning_active_at_finish = reasoning_active;

            const std::string trailing_flush = assembler.finish();
            // Every byte already delivered by release_resolved_prefix()
            // above is reflected here; the final stop_match.text_byte_offset
            // (if any) must be reported relative to the *whole* visible
            // stream, not just whatever remains in `pending` below.
            const std::size_t already_released_text_length = result.generated_text.size();

            // [0, suppress_from) is fully visible; the entry at
            // suppress_from (if any) is visible only up to
            // suppress_local_offset; everything after is fully suppressed.
            // Defaults to "nothing is suppressed" (max_tokens /
            // context_exhausted).
            std::size_t suppress_from = pending.size();
            std::size_t suppress_local_offset = 0;

            if (result.finish_reason == Qwen35FinishReason::eos) {
                suppress_from = pending.size() - 1;
            } else if (stop_matched && stop_kind == Qwen35StopKind::token_sequence) {
                suppress_from = pending.size() - stop_token_match_length;
            } else if (stop_matched && stop_kind == Qwen35StopKind::text_sequence) {
                std::size_t cumulative = 0;
                for (std::size_t i = 0; i < pending.size(); ++i) {
                    const std::size_t length = pending[i].raw_fragment.size();
                    if (stop_detected_text_offset < cumulative + length) {
                        suppress_from = i;
                        suppress_local_offset = stop_detected_text_offset - cumulative;
                        break;
                    }
                    cumulative += length;
                }
            }

            std::size_t visible_prefix_length = 0;
            for (std::size_t i = 0; i < suppress_from && i < pending.size(); ++i) {
                visible_prefix_length += pending[i].raw_fragment.size();
            }
            if (suppress_from < pending.size()) {
                visible_prefix_length += suppress_local_offset;
            }

            std::vector<std::uint32_t> matched_token_ids;
            for (std::size_t i = 0; i < pending.size(); ++i) {
                const PendingEntry& entry = pending[i];
                std::string final_fragment;
                if (i < suppress_from) {
                    final_fragment = entry.raw_fragment;
                } else if (i == suppress_from) {
                    final_fragment = entry.raw_fragment.substr(0, suppress_local_offset);
                }
                if (i >= suppress_from) {
                    matched_token_ids.push_back(entry.token_id);
                }
                // Phase 2E Slice 3C correction: applies independently of,
                // and after, the stop/EOS suppression above -- a confirmed
                // reasoning-boundary marker's visible contribution is always
                // suppressed, never its token_bytes or ledger entry.
                if (entry.reasoning_marker_suppressed) {
                    final_fragment.clear();
                }
                result.generated_text += final_fragment;

                if (callback) {
                    Qwen35GenerationEvent event;
                    event.token_id = entry.token_id;
                    event.position = entry.position;
                    event.generated_index = entry.generated_index;
                    event.probability = entry.probability;
                    event.candidate_count = entry.candidate_count;
                    event.special = entry.special;
                    event.eos = entry.eos;
                    event.token_bytes = entry.token_bytes;
                    event.text_fragment = std::move(final_fragment);
                    event.sequence_length = entry.sequence_length;
                    event.source = entry.source;
                    callback(event);
                }
            }
            if (suppress_from >= pending.size()) {
                result.generated_text += trailing_flush;
            }

            if (stop_matched) {
                // suppress_from indexes the (possibly already-shrunk)
                // `pending` array, not the overall generated_tokens ledger;
                // every matched entry's own generated_index is the
                // authoritative absolute position, and is always valid here
                // -- release_resolved_prefix() never releases an entry that
                // could still be part of a stop later confirms (see its
                // ambiguity-window proof in docs/PHASE_2E.md).
                Qwen35StopMatch match;
                match.kind = stop_kind;
                match.configured_index = stop_configured_index;
                match.generated_token_begin = pending[suppress_from].generated_index;
                match.generated_token_end = result.generated_tokens.size();
                match.text_byte_offset = already_released_text_length + visible_prefix_length;
                match.matched_token_ids = std::move(matched_token_ids);
                match.matched_text = stop_matched_text;
                result.stop_match = std::move(match);
            }
        }

        result.final_sequence_length = state_.sequence_length();
        const std::size_t expected_length =
            result.prompt_tokens.size() + result.generated_tokens.size();
        if (result.final_sequence_length != expected_length) {
            throw std::runtime_error(
                "Qwen3.5 generation accepted-token ledger does not match hybrid state");
        }
        // Phase 2F Slice 7: any predictions still queued when generation
        // ends (e.g. EOS/a stop/max_tokens/context_exhausted fired while
        // committing an earlier entry of a verified batch) were genuinely
        // decided by MTP but never committed -- account for them here,
        // distinct from Slice 5's own verification-time unused_suffix
        // (drafts never individually verified because an earlier one in
        // the same batch was rejected). Both are real, uncommitted
        // "unused" predictions from the caller's point of view.
        mtp_diag.unused_drafts += mtp_pending.size();
        result.mtp_diagnostics = mtp_diag;
        return result;
    } catch (...) {
        state_.reset();
        throw;
    }
}

std::string qwen35_generation_result_text(const Qwen35GenerationResult& result) {
    std::ostringstream output;
    output << "Qwen3.5 reference generation\n"
           << "finish_reason: " << qwen35_finish_reason_name(result.finish_reason) << '\n'
           << "prompt_tokens: " << result.prompt_tokens.size() << '\n'
           << "generated_tokens: " << result.generated_tokens.size() << '\n'
           << "final_sequence_length: " << result.final_sequence_length << '\n'
           << "tokens:";
    for (const Qwen35GeneratedToken& token : result.generated_tokens) {
        output << ' ' << token.token_id << '@' << token.position;
    }
    output << '\n' << "generated_text: " << result.generated_text << '\n';
    if (result.stop_match.has_value()) {
        const Qwen35StopMatch& match = *result.stop_match;
        output << "stop_match: kind=" << qwen35_stop_kind_name(match.kind)
               << " configured_index=" << match.configured_index
               << " generated_token_range=[" << match.generated_token_begin << ','
               << match.generated_token_end << ')'
               << " text_byte_offset=" << match.text_byte_offset
               << " matched_text=" << match.matched_text << '\n';
    }
    if (result.mtp_diagnostics.mtp_enabled) {
        const Qwen35MtpDiagnostics& diag = result.mtp_diagnostics;
        output << "mtp: requested_draft_depth=" << diag.requested_draft_depth
               << " draft_opportunities=" << diag.draft_opportunities
               << " drafts_proposed=" << diag.drafts_proposed
               << " drafts_accepted=" << diag.drafts_accepted
               << " drafts_rejected=" << diag.drafts_rejected
               << " unused_drafts=" << diag.unused_drafts
               << " full_accept_chains=" << diag.full_accept_chains
               << " partial_accept_chains=" << diag.partial_accept_chains
               << " zero_accept_chains=" << diag.zero_accept_chains << '\n';
    }
    return output.str();
}

std::string qwen35_generation_result_json(const Qwen35GenerationResult& result) {
    std::ostringstream output;
    output << '{'
           << "\"finish_reason\":\"" << qwen35_finish_reason_name(result.finish_reason) << "\","
           << "\"prompt_token_count\":" << result.prompt_tokens.size() << ','
           << "\"generated_token_count\":" << result.generated_tokens.size() << ','
           << "\"final_sequence_length\":" << result.final_sequence_length << ','
           << "\"prompt_tokens\":[";
    for (std::size_t index = 0; index < result.prompt_tokens.size(); ++index) {
        if (index != 0) output << ',';
        output << result.prompt_tokens[index];
    }
    output << "],\"generated_tokens\":[";
    for (std::size_t index = 0; index < result.generated_tokens.size(); ++index) {
        if (index != 0) output << ',';
        const Qwen35GeneratedToken& token = result.generated_tokens[index];
        output << '{'
               << "\"token_id\":" << token.token_id << ','
               << "\"position\":" << token.position << ','
               << "\"probability\":" << token.probability << ','
               << "\"candidate_count\":" << token.candidate_count
               << '}';
    }
    output << "],\"generated_text\":\"" << json_escape(result.generated_text) << "\","
           << "\"stop_match\":";
    if (result.stop_match.has_value()) {
        output << qwen35_stop_match_json(*result.stop_match);
    } else {
        output << "null";
    }
    output << ',' << "\"reasoning_active_at_finish\":"
           << (result.reasoning_active_at_finish ? "true" : "false") << ','
           << "\"reasoning_interventions\":[";
    for (std::size_t index = 0; index < result.reasoning_interventions.size(); ++index) {
        if (index != 0) output << ',';
        output << qwen35_reasoning_intervention_json(result.reasoning_interventions[index]);
    }
    output << "]," << "\"mtp_diagnostics\":" << qwen35_mtp_diagnostics_json(result.mtp_diagnostics)
           << '}';
    return output.str();
}

std::string qwen35_generation_event_json(const Qwen35GenerationEvent& event) {
    std::ostringstream output;
    output << '{'
           << "\"generated_index\":" << event.generated_index << ','
           << "\"token_id\":" << event.token_id << ','
           << "\"position\":" << event.position << ','
           << "\"probability\":" << event.probability << ','
           << "\"candidate_count\":" << event.candidate_count << ','
           << "\"special\":" << (event.special ? "true" : "false") << ','
           << "\"eos\":" << (event.eos ? "true" : "false") << ','
           << "\"token_bytes_hex\":\"" << hex_encode(event.token_bytes) << "\","
           << "\"text_fragment\":\"" << json_escape(event.text_fragment) << "\","
           << "\"sequence_length\":" << event.sequence_length << ','
           << "\"source\":\"" << qwen35_generation_token_source_name(event.source) << "\""
           << '}';
    return output.str();
}

std::string qwen35_stop_match_json(const Qwen35StopMatch& stop_match) {
    std::ostringstream output;
    output << '{'
           << "\"kind\":\"" << qwen35_stop_kind_name(stop_match.kind) << "\","
           << "\"configured_index\":" << stop_match.configured_index << ','
           << "\"generated_token_begin\":" << stop_match.generated_token_begin << ','
           << "\"generated_token_end\":" << stop_match.generated_token_end << ','
           << "\"text_byte_offset\":" << stop_match.text_byte_offset << ','
           << "\"matched_token_ids\":[";
    for (std::size_t index = 0; index < stop_match.matched_token_ids.size(); ++index) {
        if (index != 0) output << ',';
        output << stop_match.matched_token_ids[index];
    }
    output << "],\"matched_text\":\"" << json_escape(stop_match.matched_text) << "\"}";
    return output.str();
}

std::string qwen35_reasoning_intervention_json(const Qwen35ReasoningIntervention& intervention) {
    std::ostringstream output;
    output << '{'
           << "\"generated_index\":" << intervention.generated_index << ','
           << "\"reasoning_token_count\":" << intervention.reasoning_token_count << ','
           << "\"inspected_window_length\":" << intervention.inspected_window_length << ','
           << "\"detected_period\":" << intervention.detected_period << ','
           << "\"repeated_coverage\":" << intervention.repeated_coverage << ','
           << "\"intervention_number\":" << intervention.intervention_number << ','
           << "\"policy\":\"" << qwen35_reasoning_loop_policy_name(intervention.policy) << "\","
           << "\"closure_attempted\":" << (intervention.closure_attempted ? "true" : "false") << ','
           << "\"closure_succeeded\":" << (intervention.closure_succeeded ? "true" : "false") << ','
           << "\"failure_reason\":\"" << json_escape(intervention.failure_reason) << "\","
           << "\"validation_triggered\":"
           << (intervention.validation_triggered ? "true" : "false") << '}';
    return output.str();
}

}  // namespace oracle::runtime
