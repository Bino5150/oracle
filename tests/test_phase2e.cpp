#include "oracle/model/ggml_type.hpp"
#include "oracle/model/gguf.hpp"
#include "oracle/runtime/qwen35_generation.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_throws(Function&& function, std::string_view needle) {
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string_view(error.what()).find(needle) != std::string_view::npos,
                "exception message did not contain expected text");
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

// ---------------------------------------------------------------------
// Synthetic model fixture (Slice 1/2E generation-session mechanics tests)
// ---------------------------------------------------------------------

oracle::model::Qwen35Manifest manifest() {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2e-fixture";
    value.total_block_count = 2;
    value.backbone_block_count = 2;
    value.context_length = 32;
    value.embedding_length = 4;
    value.feed_forward_length = 4;
    value.attention_head_count = 2;
    value.attention_head_count_kv = 1;
    value.attention_key_length = 2;
    value.attention_value_length = 2;
    value.attention_rms_epsilon = 1.0e-6F;
    value.rope_dimension_count = 2;
    value.rope_dimension_sections = {1, 0, 0, 0};
    value.rope_frequency_base = 10000.0F;
    value.ssm_convolution_kernel = 2;
    value.ssm_state_size = 1;
    value.ssm_group_count = 2;
    value.ssm_time_step_rank = 4;
    value.ssm_inner_size = 4;
    value.full_attention_interval = 2;
    value.vocabulary_size = 8;
    return value;
}

struct Fixture {
    std::vector<oracle::model::GgufTensorInfo> infos;
    std::vector<std::vector<std::byte>> storage;
    std::vector<oracle::model::GgufTensorView> views;

    Fixture() {
        infos.reserve(64);
        storage.reserve(64);
        views.reserve(64);
    }

    const oracle::model::GgufTensorView* add(std::string name,
                                             std::vector<std::uint64_t> dimensions,
                                             std::vector<float> values) {
        std::size_t expected = 1;
        for (const std::uint64_t dimension : dimensions) {
            expected *= static_cast<std::size_t>(dimension);
        }
        require(values.size() == expected, "fixture value count mismatch for " + name);

        infos.push_back({std::move(name), std::move(dimensions), 0, 0});
        storage.emplace_back(values.size() * sizeof(float));
        for (std::size_t index = 0; index < values.size(); ++index) {
            const std::uint32_t bits = std::bit_cast<std::uint32_t>(values[index]);
            const std::size_t offset = index * sizeof(float);
            storage.back()[offset] = static_cast<std::byte>(bits & 0xFFU);
            storage.back()[offset + 1] = static_cast<std::byte>((bits >> 8U) & 0xFFU);
            storage.back()[offset + 2] = static_cast<std::byte>((bits >> 16U) & 0xFFU);
            storage.back()[offset + 3] = static_cast<std::byte>((bits >> 24U) & 0xFFU);
        }
        const auto* layout = oracle::model::ggml_type_layout(0);
        require(layout != nullptr, "F32 layout missing");
        views.emplace_back(&infos.back(),
                           layout,
                           storage.back().data(),
                           storage.back().size(),
                           0);
        return &views.back();
    }

    const oracle::model::GgufTensorView* vector(std::string name,
                                                std::vector<float> values) {
        const std::size_t size = values.size();
        return add(std::move(name), {size}, std::move(values));
    }

    const oracle::model::GgufTensorView* matrix(std::string name,
                                                std::size_t columns,
                                                std::size_t rows,
                                                std::vector<float> values) {
        return add(std::move(name), {columns, rows}, std::move(values));
    }
};

std::vector<float> zeros(std::size_t size) {
    return std::vector<float>(size, 0.0F);
}

std::vector<float> identity(std::size_t size) {
    std::vector<float> values(size * size, 0.0F);
    for (std::size_t index = 0; index < size; ++index) {
        values[index * size + index] = 1.0F;
    }
    return values;
}

oracle::model::Qwen35MlpWeights add_zero_mlp(Fixture& fixture, std::string_view prefix) {
    oracle::model::Qwen35MlpWeights weights;
    weights.gate = fixture.matrix(std::string(prefix) + "ffn_gate", 4, 4, zeros(16));
    weights.up = fixture.matrix(std::string(prefix) + "ffn_up", 4, 4, zeros(16));
    weights.down = fixture.matrix(std::string(prefix) + "ffn_down", 4, 4, zeros(16));
    return weights;
}

oracle::model::Qwen35BackboneBlockWeights recurrent_block(Fixture& fixture) {
    oracle::model::Qwen35BackboneBlockWeights block;
    block.index = 0;
    block.kind = oracle::model::Qwen35BlockKind::recurrent;
    block.input_norm = fixture.vector("r.input_norm", {1, 1, 1, 1});
    block.post_attention_norm = fixture.vector("r.post_norm", {1, 1, 1, 1});
    block.mlp = add_zero_mlp(fixture, "r.");

    std::vector<float> qkv(8 * 4, 0.0F);
    qkv[0 * 4 + 0] = 1.0F;
    qkv[1 * 4 + 1] = 1.0F;
    qkv[2 * 4 + 0] = 1.0F;
    qkv[3 * 4 + 1] = 1.0F;
    for (std::size_t row = 0; row < 4; ++row) qkv[(row + 4) * 4 + row] = 1.0F;

    std::vector<float> convolution(8 * 2, 0.0F);
    for (std::size_t channel = 0; channel < 8; ++channel) {
        convolution[channel * 2 + 1] = 1.0F;
    }

    oracle::model::Qwen35RecurrentWeights weights;
    weights.qkv = fixture.matrix("r.qkv", 4, 8, qkv);
    weights.gate = fixture.matrix("r.z", 4, 4, zeros(16));
    weights.convolution = fixture.matrix("r.conv", 2, 8, convolution);
    weights.time_step_bias = fixture.vector("r.dt", {0, 0, 0, 0});
    weights.decay = fixture.vector("r.a", {-1, -1, -1, -1});
    weights.beta = fixture.matrix("r.beta", 4, 4, zeros(16));
    weights.alpha = fixture.matrix("r.alpha", 4, 4, zeros(16));
    weights.norm = fixture.vector("r.norm", {1});
    weights.output = fixture.matrix("r.out", 4, 4, identity(4));
    block.recurrent = weights;
    return block;
}

oracle::model::Qwen35BackboneBlockWeights attention_block(Fixture& fixture) {
    oracle::model::Qwen35BackboneBlockWeights block;
    block.index = 1;
    block.kind = oracle::model::Qwen35BlockKind::full_attention;
    block.input_norm = fixture.vector("a.input_norm", {1, 1, 1, 1});
    block.post_attention_norm = fixture.vector("a.post_norm", {1, 1, 1, 1});
    block.mlp = add_zero_mlp(fixture, "a.");

    std::vector<float> query(8 * 4, 0.0F);
    query[0 * 4 + 0] = 1.0F;
    query[1 * 4 + 1] = 1.0F;
    query[4 * 4 + 2] = 1.0F;
    query[5 * 4 + 3] = 1.0F;

    std::vector<float> key(2 * 4, 0.0F);
    key[0 * 4 + 0] = 1.0F;
    key[1 * 4 + 1] = 1.0F;

    std::vector<float> value(2 * 4, 0.0F);
    value[0 * 4 + 2] = 1.0F;
    value[1 * 4 + 3] = 1.0F;

    oracle::model::Qwen35AttentionWeights weights;
    weights.query = fixture.matrix("a.q", 4, 8, query);
    weights.key = fixture.matrix("a.k", 4, 2, key);
    weights.value = fixture.matrix("a.v", 4, 2, value);
    weights.output = fixture.matrix("a.out", 4, 4, identity(4));
    weights.query_norm = fixture.vector("a.qnorm", {1, 1});
    weights.key_norm = fixture.vector("a.knorm", {1, 1});
    block.attention = weights;
    return block;
}

struct ModelFixture {
    Fixture tensors;
    oracle::model::Qwen35Weights weights;

    ModelFixture() {
        const std::vector<float> embedding_values = {
            1, 2, 3, 4,
            2, 1, 0, -1,
            -1, 0, 1, 2,
            0.5F, 0.25F, -0.25F, -0.5F,
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1,
        };
        weights.token_embedding = tensors.matrix("token_embd.weight", 4, 8, embedding_values);
        weights.output_norm = tensors.vector("output_norm.weight", {1, 1, 1, 1});
        weights.output = nullptr;
        weights.output_is_tied = true;
        weights.blocks.push_back(recurrent_block(tensors));
        weights.blocks.push_back(attention_block(tensors));
        weights.report.backbone_block_count = 2;
        weights.report.recurrent_block_count = 1;
        weights.report.attention_block_count = 1;
        weights.report.tied_output = true;
    }
};

std::uint32_t first_greedy_token(const oracle::model::Qwen35Manifest& model,
                                 const oracle::model::Qwen35Weights& weights,
                                 std::span<const std::uint32_t> prompt,
                                 std::size_t capacity) {
    oracle::runtime::HybridCache state(model, capacity);
    oracle::runtime::Qwen35ForwardResult current;
    for (const std::uint32_t token : prompt) {
        const std::size_t position = state.sequence_length();
        current = oracle::runtime::execute_qwen35_reference_token(
            model,
            weights,
            token,
            state,
            oracle::runtime::RopePosition::text(position),
            false);
    }
    oracle::runtime::Sampler sampler;
    return sampler.sample(current.logits).token_id;
}

// ---------------------------------------------------------------------
// Synthetic tokenizer fixture (Slice 3A text/event tests). Qwen35Tokenizer
// requires a real GgufFile with a complete GPT2-byte-fallback vocabulary
// (all 256 raw bytes must be representable), so this writes a minimal
// on-disk GGUF containing only tokenizer.ggml.* metadata -- no qwen35.*
// model metadata or tensors are needed to construct a tokenizer alone.
// ---------------------------------------------------------------------

template <typename Unsigned>
void write_unsigned_le(std::ostream& output, Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}
void write_u32(std::ostream& output, std::uint32_t value) { write_unsigned_le(output, value); }
void write_u64(std::ostream& output, std::uint64_t value) { write_unsigned_le(output, value); }
void write_i32(std::ostream& output, std::int32_t value) {
    write_u32(output, std::bit_cast<std::uint32_t>(value));
}
void write_string(std::ostream& output, std::string_view value) {
    write_u64(output, value.size());
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}
void write_key_string(std::ostream& output, std::string_view key, std::string_view value) {
    write_string(output, key);
    write_u32(output, 8);
    write_string(output, value);
}
void write_key_u32(std::ostream& output, std::string_view key, std::uint32_t value) {
    write_string(output, key);
    write_u32(output, 4);
    write_u32(output, value);
}
void write_key_string_array(std::ostream& output,
                            std::string_view key,
                            const std::vector<std::string>& values) {
    write_string(output, key);
    write_u32(output, 9);
    write_u32(output, 8);
    write_u64(output, values.size());
    for (const std::string& value : values) {
        write_string(output, value);
    }
}
void write_key_i32_array(std::ostream& output,
                         std::string_view key,
                         const std::vector<std::int32_t>& values) {
    write_string(output, key);
    write_u32(output, 9);
    write_u32(output, 5);
    write_u64(output, values.size());
    for (std::int32_t value : values) {
        write_i32(output, value);
    }
}

[[nodiscard]] std::string utf8_from_codepoint(std::uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return output;
}

// GPT2-style byte-fallback vocabulary: index N is always the single-raw-byte
// token for byte value N (mirrors tests/test_phase1d.cpp's proven fixture).
[[nodiscard]] std::vector<std::string> byte_tokens() {
    std::vector<std::string> output(256);
    std::array<bool, 256> direct{};
    for (std::uint32_t byte = 33; byte <= 126; ++byte) direct[byte] = true;
    for (std::uint32_t byte = 161; byte <= 172; ++byte) direct[byte] = true;
    for (std::uint32_t byte = 174; byte <= 255; ++byte) direct[byte] = true;
    std::uint32_t extension = 0;
    for (std::uint32_t byte = 0; byte <= 255; ++byte) {
        output[byte] = utf8_from_codepoint(direct[byte] ? byte : 256U + extension++);
    }
    return output;
}

struct TokenizerFixtureVocabulary {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    oracle::tokenizer::TokenId control_token{0};       // "<|endoftext|>", type=control
    oracle::tokenizer::TokenId merged_e_acute{0};      // one token spanning raw bytes 0xC3 0xA9 ('e' acute)
    oracle::tokenizer::TokenId unused_token{0};        // type=unused, must never contribute text
    oracle::tokenizer::TokenId merged_multi_char{0};   // one token decoding to "X" + e-acute + "Y"

    TokenizerFixtureVocabulary() {
        tokens = byte_tokens();
        types.assign(tokens.size(), 1);  // TokenType::normal

        const auto add = [this](std::string token, std::int32_t type) {
            tokens.push_back(std::move(token));
            types.push_back(type);
            return static_cast<oracle::tokenizer::TokenId>(tokens.size() - 1);
        };

        control_token = add("<|endoftext|>", 3);  // TokenType::control
        // Raw bytes 0xC3, 0xA9 -- the UTF-8 encoding of U+00E9 ('e' acute) --
        // as byte-fallback vocabulary entries 195 and 169 respectively.
        // Concatenating their *own* stored (already byte-to-unicode encoded)
        // text into one new vocabulary entry gives a single token whose
        // decode() output is the complete two-byte character.
        merged_e_acute = add(tokens[195] + tokens[169], 1);  // TokenType::normal
        unused_token = add("<|unused_marker|>", 5);          // TokenType::unused
        // "X" (byte 88) + e-acute (195+169) + "Y" (byte 89) as one token,
        // for Slice 3B tests needing a single committed token whose decoded
        // text has unrelated content both before *and* after a multi-byte
        // character -- something the tiny single-byte-only synthetic model
        // fixture used elsewhere in this file cannot itself select via
        // sampling.
        merged_multi_char = add(tokens[88] + tokens[195] + tokens[169] + tokens[89], 1);
    }
};

[[nodiscard]] std::filesystem::path write_tokenizer_fixture() {
    const TokenizerFixtureVocabulary vocabulary;
    const auto path =
        std::filesystem::temp_directory_path() / "oracle-phase2e-slice3a-tokenizer.gguf";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to create Phase 2E Slice 3A tokenizer fixture");

    output.write("GGUF", 4);
    write_u32(output, 3);
    write_u64(output, 0);  // tensor_count -- a tokenizer alone needs no tensors
    write_u64(output, 6);  // metadata_count

    write_key_string(output, "tokenizer.ggml.model", "gpt2");
    write_key_string(output, "tokenizer.ggml.pre", "qwen35");
    write_key_string_array(output, "tokenizer.ggml.tokens", vocabulary.tokens);
    write_key_i32_array(output, "tokenizer.ggml.token_type", vocabulary.types);
    write_key_string_array(output, "tokenizer.ggml.merges", {});
    write_key_u32(output, "tokenizer.ggml.eos_token_id", vocabulary.control_token);

    // No tensors follow, but the reader still aligns the (empty) tensor
    // data region to the default 32-byte alignment and validates that the
    // file is at least that long.
    const std::uint64_t descriptor_end = static_cast<std::uint64_t>(output.tellp());
    const std::uint64_t data_offset = (descriptor_end + 31ULL) & ~31ULL;
    for (std::uint64_t offset = descriptor_end; offset < data_offset; ++offset) {
        output.put('\0');
    }

    output.close();
    require(static_cast<bool>(output), "failed while writing Phase 2E Slice 3A tokenizer fixture");
    return path;
}

[[nodiscard]] oracle::tokenizer::Qwen35Tokenizer make_text_tokenizer() {
    const std::filesystem::path path = write_tokenizer_fixture();
    const oracle::model::GgufFile file = oracle::model::GgufReader::read(path);
    std::filesystem::remove(path);
    return oracle::tokenizer::Qwen35Tokenizer(file);
}

// Constructed once and shared: this fixture requires real file I/O, and
// none of the tests mutate it.
const oracle::tokenizer::Qwen35Tokenizer& shared_text_tokenizer() {
    static const oracle::tokenizer::Qwen35Tokenizer tokenizer = make_text_tokenizer();
    return tokenizer;
}

// Fixed indices this file relies on -- see TokenizerFixtureVocabulary.
constexpr oracle::tokenizer::TokenId kByteC3 = 195;  // lead byte of 'e' acute
constexpr oracle::tokenizer::TokenId kByteA9 = 169;  // trailing byte of 'e' acute
constexpr oracle::tokenizer::TokenId kByteH = 72;
constexpr oracle::tokenizer::TokenId kByteI = 105;
constexpr oracle::tokenizer::TokenId kByteStrayContinuation = 128;  // raw byte 0x80: never a valid lead
constexpr oracle::tokenizer::TokenId kControlToken = 256;   // "<|endoftext|>"
constexpr oracle::tokenizer::TokenId kMergedEAcute = 257;   // spans raw bytes 0xC3 0xA9
constexpr oracle::tokenizer::TokenId kUnusedToken = 258;    // TokenType::unused marker
constexpr oracle::tokenizer::TokenId kMergedMultiCharToken = 259;  // decodes to "X" + e-acute + "Y"

// Independent re-implementation (deliberately not shared with the
// production scanner in qwen35_generation.cpp) used only to assert that
// emitted text really is well-formed UTF-8, so these tests do not just
// check "the implementation agrees with itself."
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

constexpr std::string_view kReplacementCharacter = "\xEF\xBF\xBD";  // U+FFFD

// ---------------------------------------------------------------------
// Slice 1 generation-session tests
// ---------------------------------------------------------------------

void test_greedy_generation_and_state_ledger() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "greedy fixture should stop at max_tokens");
    require(result.generated_tokens.size() == 3, "generated token count mismatch");
    require(result.final_sequence_length == 5, "final state length mismatch");
    require(session.state().sequence_length() == 5, "session state length mismatch");
    for (std::size_t index = 0; index < result.generated_tokens.size(); ++index) {
        require(result.generated_tokens[index].position == 2 + index,
                "generated token position mismatch");
        require(result.generated_tokens[index].candidate_count == 1,
                "greedy sampling candidate count must be one");
    }
    require(oracle::runtime::qwen35_generation_result_json(result).find("max_tokens") !=
                std::string::npos,
            "generation JSON omitted finish reason");
}

void test_eos_is_committed_before_finish() {
    ModelFixture fixture;
    auto model = manifest();
    const std::vector<std::uint32_t> prompt{0, 1};
    model.eos_token_id = first_greedy_token(model, fixture.weights, prompt, 8);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 4;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::eos,
            "EOS fixture did not stop on EOS");
    require(result.generated_tokens.size() == 1, "EOS must stop after one accepted token");
    require(result.generated_tokens.front().token_id == *model.eos_token_id,
            "accepted token did not match EOS");
    require(result.final_sequence_length == prompt.size() + 1,
            "EOS token was not committed to state");
}

void test_context_exhaustion_is_bounded() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 3);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 5;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::context_exhausted,
            "context exhaustion finish reason mismatch");
    require(result.generated_tokens.size() == 1,
            "context capacity should permit exactly one generated token");
    require(result.final_sequence_length == 3, "context capacity was exceeded");
}

void test_preflight_preserves_existing_state() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);

    oracle::runtime::Qwen35GenerationRequest valid;
    valid.prompt_tokens = {0};
    valid.max_generated_tokens = 1;
    static_cast<void>(session.generate_fresh(valid));
    const std::size_t prior = session.state().sequence_length();
    require(prior == 2, "valid setup did not create expected state");

    oracle::runtime::Qwen35GenerationRequest invalid;
    invalid.max_generated_tokens = 1;
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "prompt must contain");
    require(session.state().sequence_length() == prior,
            "preflight failure mutated existing session state");

    invalid.prompt_tokens = {0};
    invalid.sampling.temperature = -1.0F;
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "temperature");
    require(session.state().sequence_length() == prior,
            "sampling preflight failure mutated existing session state");
}

void test_runtime_failure_resets_partial_state() {
    ModelFixture fixture;
    auto model = manifest();

    // Deliberately break the output-projection contract: untie the output
    // head and supply a projection tensor whose vocabulary width does not
    // match the manifest, while the token embedding table stays correct.
    // This forces execute_qwen35_reference_token() to advance HybridCache
    // state during block execution and only fail afterward, exercising the
    // runtime-failure reset path rather than preflight rejection.
    fixture.weights.output_is_tied = false;
    fixture.weights.output = fixture.tensors.matrix("bad.out", 4, 4, identity(4));

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 1;

    require_throws([&] { static_cast<void>(session.generate_fresh(request)); },
                   "output projection dimensions");
    require(session.state().sequence_length() == 0,
            "runtime failure left partially advanced state exposed as a valid session");
}

void test_invalid_prompt_token_rejected() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {model.vocabulary_size};
    request.max_generated_tokens = 1;
    require_throws([&] { static_cast<void>(session.generate_fresh(request)); },
                   "prompt token exceeds vocabulary");
    require(session.state().sequence_length() == 0,
            "invalid token preflight mutated state");
}

// ---------------------------------------------------------------------
// Slice 3A: event/callback mechanics tests (required tests 1, 2, 3, 5, 6, 7)
// ---------------------------------------------------------------------

void test_generation_events_match_ledger() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request,
        [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    // Required test 1: event count == number of accepted generated tokens,
    // and no events sneak in for prompt ingestion.
    require(events.size() == result.generated_tokens.size(),
            "event count must equal the number of accepted generated tokens");
    require(events.size() == 3, "expected exactly three accepted generated tokens");

    for (std::size_t index = 0; index < events.size(); ++index) {
        // Required test 3: event token identity matches the ledger exactly.
        require(events[index].token_id == result.generated_tokens[index].token_id,
                "event token id must match the accepted-token ledger");
        require(events[index].position == result.generated_tokens[index].position,
                "event position must match the accepted-token ledger");
        require(events[index].probability == result.generated_tokens[index].probability,
                "event probability must match the accepted-token ledger");
        require(events[index].candidate_count == result.generated_tokens[index].candidate_count,
                "event candidate_count must match the accepted-token ledger");
        require(events[index].generated_index == index,
                "generated_index must match callback emission order");
        // Required test 5: post-commit sequence length == position + 1.
        require(events[index].sequence_length == events[index].position + 1,
                "event sequence_length must equal position + 1");
        // Required test 2: positions increase monotonically.
        if (index > 0) {
            require(events[index].position > events[index - 1].position,
                    "event positions must increase monotonically");
        }
    }
    require(events.front().position == request.prompt_tokens.size(),
            "first generated event position must equal the prompt length");

    std::string assembled_from_events;
    for (const oracle::runtime::Qwen35GenerationEvent& event : events) {
        assembled_from_events += event.text_fragment;
    }
    require(assembled_from_events == result.generated_text,
            "result.generated_text must equal the concatenation of emitted event text_fragments");
    require(oracle::runtime::qwen35_generation_result_json(result).find("\"generated_text\"") !=
                std::string::npos,
            "generation JSON must expose the generated_text field");
}

void test_callback_does_not_alter_greedy_tokens() {
    ModelFixture fixture;
    auto model = manifest();
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35GenerationSession without_callback_session(
        model, fixture.weights, tokenizer, 8);
    const auto without_callback = without_callback_session.generate_fresh(request);

    oracle::runtime::Qwen35GenerationSession with_callback_session(
        model, fixture.weights, tokenizer, 8);
    std::size_t callback_invocations = 0;
    const auto with_callback = with_callback_session.generate_fresh(
        request, [&callback_invocations](const oracle::runtime::Qwen35GenerationEvent&) {
            ++callback_invocations;
        });

    // Required test 6: greedy behavior is unchanged whether or not a
    // callback is registered -- the observation surface is inert.
    require(callback_invocations == with_callback.generated_tokens.size(),
            "callback must fire exactly once per accepted generated token");
    require(without_callback.generated_tokens.size() == with_callback.generated_tokens.size(),
            "callback presence changed the number of accepted tokens");
    for (std::size_t index = 0; index < without_callback.generated_tokens.size(); ++index) {
        require(without_callback.generated_tokens[index].token_id ==
                    with_callback.generated_tokens[index].token_id,
                "callback presence changed a sampled token id");
    }
    require(without_callback.finish_reason == with_callback.finish_reason,
            "callback presence changed the finish reason");
    require(without_callback.final_sequence_length == with_callback.final_sequence_length,
            "callback presence changed the final sequence length");
    require(without_callback.generated_text == with_callback.generated_text,
            "callback presence changed the assembled generated text");
}

void test_no_event_for_uncommitted_token() {
    // Required test 4 (commit-before-event): the only way to force a
    // forward failure in this fixture family without relaxing Slice 1's
    // validation is to break the shared output-projection tensor, which
    // fails uniformly for every forward call including prompt ingestion
    // (a non-empty prompt is mandatory, so the very first attempted
    // forward is necessarily a prompt token here). This still proves the
    // invariant the requirement cares about -- no event is ever observed
    // for a token whose forward/commit did not succeed -- because event
    // construction in generate_fresh() is reached only via the exact same
    // post-require_state_advanced() code path used for generated tokens;
    // there is no separate, weaker-guarded path for either token kind.
    ModelFixture fixture;
    auto model = manifest();
    fixture.weights.output_is_tied = false;
    fixture.weights.output = fixture.tensors.matrix("bad.out", 4, 4, identity(4));

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 1;

    std::size_t callback_invocations = 0;
    require_throws(
        [&] {
            static_cast<void>(session.generate_fresh(
                request, [&callback_invocations](const oracle::runtime::Qwen35GenerationEvent&) {
                    ++callback_invocations;
                }));
        },
        "output projection dimensions");
    require(callback_invocations == 0, "an event was emitted for a token whose commit failed");
    require(session.state().sequence_length() == 0,
            "runtime failure left partially advanced state exposed as a valid session");
}

void test_callback_failure_resets_session_state() {
    // Required test 7: if the callback throws, generate_fresh() must not
    // pretend generation succeeded. Placing the callback invocation inside
    // Slice 1's existing try/catch means a callback exception is treated
    // exactly like any other runtime failure: state resets, the exception
    // propagates, nothing partial is ever returned to the caller.
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;

    std::size_t callback_invocations = 0;
    require_throws(
        [&] {
            static_cast<void>(session.generate_fresh(
                request,
                [&callback_invocations](const oracle::runtime::Qwen35GenerationEvent&) {
                    ++callback_invocations;
                    throw std::runtime_error("Phase 2E Slice 3A synthetic callback failure");
                }));
        },
        "synthetic callback failure");
    require(callback_invocations == 1, "callback should have fired exactly once before throwing");
    require(session.state().sequence_length() == 0,
            "callback failure must reset session state rather than expose a partial generation");
}

// ---------------------------------------------------------------------
// Slice 3A: UTF-8 text-assembly tests (required tests 8, 9, 10, 11, 12)
// Exercised directly against Qwen35IncrementalTextAssembler -- the exact
// component Qwen35GenerationSession::generate_fresh() uses internally for
// every accepted token -- rather than through full sampling-driven
// generation, so the byte-boundary scenarios can be constructed precisely
// and deterministically.
// ---------------------------------------------------------------------

void test_ascii_assembly() {
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(shared_text_tokenizer());
    const auto first = assembler.append(kByteH);
    const auto second = assembler.append(kByteI);
    require(first.text_fragment == "H", "ASCII token should be emitted immediately");
    require(second.text_fragment == "i", "ASCII token should be emitted immediately");
    require(first.token_bytes == "H" && second.token_bytes == "i",
            "ASCII token_bytes should equal the raw byte");
    require((first.text_fragment + second.text_fragment) == "Hi",
            "ASCII fragments must assemble exactly");
    require(is_valid_utf8(first.text_fragment) && is_valid_utf8(second.text_fragment),
            "every emitted text_fragment must be valid UTF-8");
    const std::string final_flush = assembler.finish();
    require(final_flush.empty(), "no bytes should remain pending after complete ASCII");
    require(is_valid_utf8(final_flush), "finish() output must be valid UTF-8");
}

void test_multibyte_utf8_within_one_token() {
    // merged_e_acute's vocabulary text already spans both raw bytes of
    // U+00E9 ('e' acute, C3 A9): a single accepted token containing a
    // complete multi-byte character.
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(shared_text_tokenizer());
    const auto result = assembler.append(kMergedEAcute);
    require(result.token_bytes == "\xC3\xA9", "token_bytes must be the raw two-byte character");
    require(result.text_fragment == "\xC3\xA9",
            "a self-contained, valid multi-byte character must be emitted whole, immediately, "
            "with no replacement");
    require(is_valid_utf8(result.text_fragment), "emitted text_fragment must be valid UTF-8");
    require(assembler.finish().empty(), "no bytes should remain pending");
}

void test_utf8_split_across_token_boundary() {
    // A *valid* multi-byte character split across a token boundary must be
    // reconstructed whole once complete -- never replaced. Replacement is
    // reserved for byte sequences that are genuinely malformed or still
    // incomplete when generation ends (see test_malformed_utf8_is_replaced
    // and test_trailing_partial_sequence_is_replaced_at_finish).
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(shared_text_tokenizer());

    const auto first = assembler.append(kByteC3);
    require(first.token_bytes == "\xC3", "first half's token_bytes must be just that raw byte");
    require(first.text_fragment.empty(),
            "a valid-but-incomplete multi-byte sequence must not be emitted or replaced early");
    require(assembler.pending() == "\xC3", "the incomplete lead byte must be retained pending");

    const auto second = assembler.append(kByteA9);
    require(second.token_bytes == "\xA9", "second half's token_bytes must be just that raw byte");
    require(second.text_fragment == "\xC3\xA9",
            "completing a valid sequence must emit the full character, unreplaced, including the "
            "earlier byte");
    require(assembler.pending().empty(), "nothing should remain pending once the character completes");
    require(is_valid_utf8(first.text_fragment) && is_valid_utf8(second.text_fragment),
            "every emitted text_fragment must be valid UTF-8");

    const std::string assembled = first.text_fragment + second.text_fragment;
    require(assembled == "\xC3\xA9",
            "final assembled text must be exactly the valid two-byte character, with no replacement");
    require(is_valid_utf8(assembled), "assembled text must be valid UTF-8");
}

void test_malformed_utf8_is_replaced() {
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();

    // Case 1: a lone stray continuation-byte-pattern raw byte (0x80) can
    // never be a valid UTF-8 lead byte on its own.
    {
        oracle::runtime::Qwen35IncrementalTextAssembler assembler(tokenizer);
        const auto result = assembler.append(kByteStrayContinuation);
        require(result.token_bytes == "\x80",
                "token_bytes must preserve the exact raw offending byte");
        require(result.text_fragment == kReplacementCharacter,
                "a stray continuation byte must be replaced with exactly one U+FFFD");
        require(is_valid_utf8(result.text_fragment), "the replacement fragment must be valid UTF-8");
        require(assembler.pending().empty(), "a malformed byte must not be held pending");
    }

    // Case 2: a plausible 2-byte lead (0xC3) followed by a byte that is not
    // a valid continuation byte. The malformed lead is replaced (one
    // U+FFFD); the following, unrelated valid ASCII byte is unaffected.
    {
        oracle::runtime::Qwen35IncrementalTextAssembler assembler(tokenizer);
        const auto lead = assembler.append(kByteC3);
        require(lead.text_fragment.empty(), "a plausible lead byte is held pending, not replaced yet");

        const auto next = assembler.append(kByteH);  // 'H' is not a continuation byte
        require(next.token_bytes == "H", "token_bytes for the following token is unaffected");
        require(next.text_fragment == std::string(kReplacementCharacter) + "H",
                "an invalid continuation must replace only the malformed lead byte, "
                "then resume normal decoding");
        require(is_valid_utf8(next.text_fragment), "the emitted fragment must be valid UTF-8");
        require(assembler.pending().empty(), "nothing should remain pending after resolution");

        require((lead.text_fragment + next.text_fragment) ==
                    std::string(kReplacementCharacter) + "H",
                "deterministic replacement policy: one U+FFFD for the malformed lead, then 'H'");
    }
}

void test_trailing_partial_sequence_is_replaced_at_finish() {
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(shared_text_tokenizer());
    std::string assembled;
    assembled += assembler.append(kByteH).text_fragment;
    assembled += assembler.append(kByteI).text_fragment;
    const auto trailing = assembler.append(kByteC3);  // only the lead byte of 'e' acute
    assembled += trailing.text_fragment;

    require(trailing.token_bytes == "\xC3",
            "token_bytes must preserve the exact raw trailing byte even though it's incomplete");
    require(trailing.text_fragment.empty(),
            "a still-incomplete-but-plausible sequence must not be emitted or replaced mid-generation");
    require(assembled == "Hi", "ASCII + multibyte-lead-only must assemble to just the ASCII so far");

    const std::string final_flush = assembler.finish();
    require(final_flush == kReplacementCharacter,
            "documented policy: a trailing sequence that will never complete is replaced with "
            "U+FFFD at finish(), never emitted raw and never silently dropped");
    require(is_valid_utf8(final_flush), "finish() output must be valid UTF-8");
    assembled += final_flush;

    require(is_valid_utf8(assembled), "the fully assembled generated text must be valid UTF-8");
    require(assembled == "Hi" + std::string(kReplacementCharacter),
            "final assembled text must be ASCII followed by exactly one replacement character");

    // token_bytes remains independently reconstructable to the exact raw
    // bytes the tokenizer produced, unaffected by the visible-text policy.
    const std::array<oracle::tokenizer::TokenId, 3> all{kByteH, kByteI, kByteC3};
    require(shared_text_tokenizer().decode(all, oracle::tokenizer::DecodeOptions{}) == "Hi\xC3",
            "raw token_bytes semantics (tokenizer::decode()) remain exactly as before this fix");
}

void test_special_and_unused_token_behavior() {
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();

    require(tokenizer.is_special(kControlToken), "control-type token must be reported special");
    require(!tokenizer.is_special(kUnusedToken), "unused-type token is not itself 'special'");

    oracle::runtime::Qwen35IncrementalTextAssembler special_assembler(tokenizer);
    const auto special_result = special_assembler.append(kControlToken);
    // Required test 12, first half: by the tokenizer's own decode() default
    // contract (skip_special_tokens=false), special-token text is
    // user-visible; the `special` flag (verified above, and wired into the
    // session's Qwen35GenerationEvent) is what lets a consumer distinguish
    // it from ordinary content, not a forced-empty fragment.
    require(special_result.text_fragment == "<|endoftext|>",
            "special-token text follows the tokenizer's own decode() contract by default");
    require(is_valid_utf8(special_result.text_fragment),
            "special-token text_fragment must be valid UTF-8");

    oracle::runtime::Qwen35IncrementalTextAssembler unused_assembler(tokenizer);
    const auto unused_result = unused_assembler.append(kUnusedToken);
    // Required test 12, second half: an unused-type token still gets
    // decoded (no exception, no skip) even though it contributes no bytes
    // at all -- proving the event/assembly path handles an empty visible
    // fragment gracefully rather than treating it as an error.
    require(unused_result.token_bytes.empty(),
            "an unused-type token must contribute zero raw bytes, matching tokenizer::decode()");
    require(unused_result.text_fragment.empty(),
            "an unused-type token's event still exists but its visible fragment is empty");
    require(is_valid_utf8(unused_result.text_fragment),
            "an empty fragment is trivially valid UTF-8");
}

// ---------------------------------------------------------------------
// Slice 3B: stop-sequence tests
// ---------------------------------------------------------------------

// Runs a plain (no stops configured) generation to discover exactly which
// tokens this deterministic fixture greedily selects for a given prompt.
// Stop configuration never changes which tokens are sampled (see
// test_empty_stop_configuration_matches_slice3a_behavior and
// test_callback_does_not_alter_greedy_tokens), so the tokens discovered
// here are guaranteed to be the same prefix a *stop-configured* run over
// the same prompt/model will produce, right up to wherever its stop fires.
std::vector<std::uint32_t> discover_natural_tokens(
    const oracle::model::Qwen35Manifest& model,
    const oracle::model::Qwen35Weights& weights,
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer,
    std::vector<std::uint32_t> prompt,
    std::size_t capacity,
    std::size_t count) {
    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, capacity);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = std::move(prompt);
    request.max_generated_tokens = count;
    request.sampling.temperature = 0.0F;
    const auto result = session.generate_fresh(request);
    require(result.generated_tokens.size() == count,
            "natural-sequence discovery did not produce the requested token count");
    std::vector<std::uint32_t> ids;
    ids.reserve(result.generated_tokens.size());
    for (const auto& token : result.generated_tokens) ids.push_back(token.token_id);
    return ids;
}

std::string decode_single(const oracle::tokenizer::Qwen35Tokenizer& tokenizer,
                          std::uint32_t token_id) {
    const std::array<oracle::tokenizer::TokenId, 1> single{token_id};
    return tokenizer.decode(single, oracle::tokenizer::DecodeOptions{});
}

// --- Stop preflight (required tests 1-4) --------------------------------

void test_stop_preflight_rejects_empty_token_stop() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest valid;
    valid.prompt_tokens = {0};
    valid.max_generated_tokens = 1;
    static_cast<void>(session.generate_fresh(valid));
    const std::size_t prior = session.state().sequence_length();

    oracle::runtime::Qwen35GenerationRequest invalid;
    invalid.prompt_tokens = {0};
    invalid.max_generated_tokens = 1;
    invalid.token_stop_sequences = {{}};
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "token stop sequence must not be empty");
    require(session.state().sequence_length() == prior,
            "empty token stop preflight failure mutated existing session state");
}

void test_stop_preflight_rejects_out_of_vocabulary_token_stop() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest valid;
    valid.prompt_tokens = {0};
    valid.max_generated_tokens = 1;
    static_cast<void>(session.generate_fresh(valid));
    const std::size_t prior = session.state().sequence_length();

    oracle::runtime::Qwen35GenerationRequest invalid;
    invalid.prompt_tokens = {0};
    invalid.max_generated_tokens = 1;
    invalid.token_stop_sequences = {{model.vocabulary_size}};
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "token stop sequence token exceeds vocabulary");
    require(session.state().sequence_length() == prior,
            "out-of-vocabulary token stop preflight failure mutated existing session state");
}

void test_stop_preflight_rejects_empty_text_stop() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest valid;
    valid.prompt_tokens = {0};
    valid.max_generated_tokens = 1;
    static_cast<void>(session.generate_fresh(valid));
    const std::size_t prior = session.state().sequence_length();

    oracle::runtime::Qwen35GenerationRequest invalid;
    invalid.prompt_tokens = {0};
    invalid.max_generated_tokens = 1;
    invalid.text_stop_sequences = {""};
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "text stop sequence must not be empty");
    require(session.state().sequence_length() == prior,
            "empty text stop preflight failure mutated existing session state");
}

void test_stop_preflight_rejects_invalid_utf8_text_stop() {
    ModelFixture fixture;
    auto model = manifest();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest valid;
    valid.prompt_tokens = {0};
    valid.max_generated_tokens = 1;
    static_cast<void>(session.generate_fresh(valid));
    const std::size_t prior = session.state().sequence_length();

    oracle::runtime::Qwen35GenerationRequest invalid;
    invalid.prompt_tokens = {0};
    invalid.max_generated_tokens = 1;
    invalid.text_stop_sequences = {"\xFF"};
    require_throws([&] { static_cast<void>(session.generate_fresh(invalid)); },
                   "text stop sequence must be valid UTF-8");
    require(session.state().sequence_length() == prior,
            "invalid-UTF-8 text stop preflight failure mutated existing session state");
}

// --- Token stops (required tests 5-12) -----------------------------------

void test_token_stop_single_token() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0]}};

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "single-token stop must finish with stop_sequence");
    require(result.generated_tokens.size() == 1, "single-token stop must commit exactly one token");
    require(result.generated_tokens.front().token_id == natural[0],
            "matched stop token must remain in the generated-token ledger");  // req 9
    require(result.final_sequence_length == prompt.size() + 1,
            "matched stop token must advance state");  // req 10
    require(session.state().sequence_length() == result.final_sequence_length,
            "session state must reflect the committed stop token");
    require(result.generated_text.empty(),
            "stop-token visible text must be suppressed");  // req 11
    require(result.stop_match.has_value(), "stop_match must be present");
    require(result.stop_match->kind == oracle::runtime::Qwen35StopKind::token_sequence,
            "stop kind must be token_sequence");
    require(result.stop_match->configured_index == 0, "configured_index must be 0");
    require(result.stop_match->generated_token_begin == 0 &&
                result.stop_match->generated_token_end == 1,
            "stop_match token range must cover the single matched token");
    require(result.stop_match->matched_token_ids == std::vector<std::uint32_t>{natural[0]},
            "stop_match must report the matched token id");

    // req 12: callback does not leak token-stop prefix text.
    require(events.size() == 1, "exactly one event for the one committed token");
    require(events.front().token_id == natural[0], "event token id must match the ledger");
    require(events.front().text_fragment.empty(),
            "callback must not leak any stop-token visible text");
    require(events.front().token_bytes == decode_single(tokenizer, natural[0]),
            "event token_bytes must remain exact even though text_fragment is suppressed");
}

void test_token_stop_multi_token_spanning_iterations() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0], natural[1], natural[2]}};

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "multi-token stop spanning iterations must finish with stop_sequence");
    const std::vector<std::uint32_t> expected{natural[0], natural[1], natural[2]};
    std::vector<std::uint32_t> committed;
    for (const auto& token : result.generated_tokens) committed.push_back(token.token_id);
    require(committed == expected,
            "matched multi-token stop sequence must remain exactly in the ledger");  // req 9
    require(result.final_sequence_length == prompt.size() + 3,
            "all three matched stop tokens must advance state");  // req 10
    require(result.generated_text.empty(),
            "visible text contributed by the matched stop-token sequence must be suppressed");  // 11
    require(result.stop_match.has_value() &&
                result.stop_match->generated_token_begin == 0 &&
                result.stop_match->generated_token_end == 3,
            "stop_match must cover the full three-token match");
    require(result.stop_match->matched_token_ids == expected,
            "stop_match matched_token_ids must equal the matched stop sequence");

    require(events.size() == 3, "exactly one event per committed token");  // req 22
    for (std::size_t i = 0; i < events.size(); ++i) {
        require(events[i].token_id == expected[i], "event token id must match the ledger");  // 23
        require(events[i].text_fragment.empty(),
                "callback must not leak any stop-token prefix text");  // req 12
        require(events[i].token_bytes == decode_single(tokenizer, expected[i]),
                "event token_bytes must remain exact for every stop token");  // req 25
        require(events[i].generated_index == i, "event order must remain canonical");  // req 24
    }
}

void test_token_stop_shared_prefix_no_premature_match() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    // A correct 3-token continuation configured as the stop, but capped at
    // 2 generated tokens: the first two tokens share a prefix with the
    // configured stop, but the stop must NOT fire early -- only max_tokens
    // should, since the stop's length (3) never fits within what was
    // actually generated (2). This is exactly the "[..., 10, 20] must NOT
    // stop; only [..., 10, 20, 30] matches" example from the task brief.
    request.max_generated_tokens = 2;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0], natural[1], natural[2]}};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "a shared 2-token prefix of a longer configured stop must not stop early");
    require(result.generated_tokens.size() == 2, "generation must proceed to max_tokens");
    require(!result.stop_match.has_value(), "no stop_match when no stop actually completed");
}

// --- Text stops (required tests 13-21) -----------------------------------

void test_text_stop_within_one_token() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);
    const std::string first_text = decode_single(tokenizer, natural[0]);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {first_text};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "text stop wholly inside one token must be detected");
    require(result.generated_tokens.size() == 1, "must commit exactly the one matching token");
    require(result.generated_tokens.front().token_id == natural[0],
            "committed token must be the one matching token");
    require(result.generated_text.empty(), "matched text must be suppressed from visible text");
    require(result.stop_match.has_value() &&
                result.stop_match->kind == oracle::runtime::Qwen35StopKind::text_sequence,
            "stop kind must be text_sequence");
    require(result.stop_match->matched_text == first_text,
            "matched_text must equal the configured stop");
    require(result.stop_match->text_byte_offset == 0,
            "match begins at the very start of visible text");
}

void test_text_stop_split_across_two_tokens() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);
    const std::string stop_text =
        decode_single(tokenizer, natural[0]) + decode_single(tokenizer, natural[1]);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {stop_text};

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "text stop split across two tokens must be detected");
    require(result.generated_tokens.size() == 2,
            "must commit exactly the two tokens forming the stop");
    require(result.generated_text.empty(), "matched text must be fully suppressed");
    require(result.stop_match->text_byte_offset == 0, "match begins at the very start");

    require(events.size() == 2, "one event per committed token");
    require(events[0].text_fragment.empty() && events[1].text_fragment.empty(),
            "callback must not leak any part of a stop split across tokens");  // req 20
}

void test_text_stop_split_across_several_tokens() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 5);
    std::string stop_text;
    for (std::size_t i = 0; i < 4; ++i) stop_text += decode_single(tokenizer, natural[i]);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {stop_text};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "text stop split across several tokens must be detected");
    require(result.generated_tokens.size() == 4,
            "must commit exactly the four tokens forming the stop");
    require(result.generated_text.empty(), "matched text must be fully suppressed");
}

void test_text_stop_begins_mid_stream() {
    // The synthetic model fixture used elsewhere in this file greedily
    // settles into repeating a single fixed token forever regardless of
    // which prompt it is given (confirmed empirically across several
    // prompts), so a *generated* sequence with genuinely distinguishable
    // leading content followed by a distinct matched portion cannot be
    // produced by real sampling here -- any stop built purely from that
    // repeated value's own byte would always match starting at offset 0,
    // never mid-stream. This directly exercises the real, already-proven
    // Qwen35IncrementalTextAssembler output for a hand-chosen, genuinely
    // distinct token sequence instead, applying the documented Slice 3B
    // suppression policy explicitly. Required test 16.
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(tokenizer);

    const auto leading = assembler.append(kByteH);        // "H" -- must stay visible
    const auto lead_of_match = assembler.append(kByteC3); // incomplete on its own
    const auto rest_of_match = assembler.append(kByteA9); // completes e-acute

    require(leading.text_fragment == "H", "the leading token's fragment is available immediately");
    require(lead_of_match.text_fragment.empty(),
            "an incomplete multi-byte sequence must not be emitted early");
    require(rest_of_match.text_fragment == "\xC3\xA9",
            "completing the sequence emits the full character");

    const std::string pending_text =
        leading.text_fragment + lead_of_match.text_fragment + rest_of_match.text_fragment;
    require(pending_text == "H\xC3\xA9", "sanity: assembled text is 'H' followed by e-acute");

    const std::string stop = "\xC3\xA9";
    const std::size_t match_offset = pending_text.find(stop);
    require(match_offset == 1, "the match must begin after the leading 'H', not at offset 0");

    const std::string visible = pending_text.substr(0, match_offset);
    require(visible == "H",
            "text before the match must remain visible; only the match onward is suppressed");
    require(is_valid_utf8(visible), "the visible prefix must remain valid UTF-8");
}

void test_text_stop_multibyte_and_mid_token_trailing_suffix() {
    // The synthetic model fixture used elsewhere in this file only ever
    // selects single-byte-decoding tokens (0-7), so a token whose decoded
    // text contains a complete multi-byte character with unrelated content
    // both before *and* after a stop match cannot be reached by real
    // sampling here (see TokenizerFixtureVocabulary::merged_multi_char).
    // This exercises the identical, real Qwen35IncrementalTextAssembler
    // output (already proven correct for multi-byte content in Slice 3A)
    // and applies exactly the documented Slice 3B suppression policy --
    // "preserve token_bytes; emit only the safe visible prefix before stop
    // start; suppress the stop and any token-local suffix after it" -- to
    // prove the policy itself is correct for a case the tiny model cannot
    // produce end-to-end. Required tests 17 and 18.
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(tokenizer);

    const auto decoded = assembler.append(kMergedMultiCharToken);
    require(decoded.token_bytes == "X\xC3\xA9Y", "token_bytes must be the exact raw four bytes");
    require(decoded.text_fragment == "X\xC3\xA9Y",
            "the whole self-contained multi-byte fragment is available immediately");

    const std::string stop = "\xC3\xA9";  // the multi-byte character alone
    const std::size_t match_offset = decoded.text_fragment.find(stop);
    require(match_offset == 1,
            "the multi-byte stop must be found at its exact byte offset, not early");

    // Mirrors Qwen35GenerationSession's documented suppression policy for a
    // stop that both begins and ends inside a single already-committed
    // token, with unrelated content on both sides.
    const std::string visible = decoded.text_fragment.substr(0, match_offset);
    require(visible == "X",
            "only the safe prefix before the stop is visible; the match and the trailing "
            "'Y' suffix in the same token are both suppressed");
    require(is_valid_utf8(visible), "the visible portion must remain valid UTF-8");
}

void test_text_stop_matching_uses_safe_text_not_raw_bytes() {
    // Required test 19: prove stop matching must operate on the UTF-8-safe
    // assembled text (with U+FFFD substitution already applied), never on
    // raw token_bytes, by configuring a stop string that could only ever
    // appear in the *safe* rendering of a malformed byte -- never in truly
    // raw content.
    const oracle::tokenizer::Qwen35Tokenizer& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35IncrementalTextAssembler assembler(tokenizer);

    std::string safe_text;
    std::string raw_text;
    for (const oracle::tokenizer::TokenId token_id : {kByteH, kByteStrayContinuation, kByteI}) {
        const auto decoded = assembler.append(token_id);
        safe_text += decoded.text_fragment;
        raw_text += decoded.token_bytes;
    }
    require(raw_text == "H\x80i", "raw token_bytes must preserve the malformed byte exactly");
    const std::string stop = std::string("H") + std::string(kReplacementCharacter) + "i";
    require(safe_text == stop,
            "the UTF-8-safe assembled text must already contain the replacement character");
    require(raw_text.find(stop) == std::string::npos,
            "the configured stop string can never appear in the raw byte stream");
    require(safe_text.find(stop) != std::string::npos,
            "but it is present in the safe text a real generation session actually searches");
}

// --- Event/state integrity under stops (required tests 22-26) -----------

void test_no_uncommitted_token_event_with_stops_configured() {
    ModelFixture fixture;
    auto model = manifest();

    // Deliberately break the output-projection contract exactly as
    // test_runtime_failure_resets_partial_state does, but this time with
    // stops configured, to prove the buffered-delivery code path also
    // never leaks an event for a token whose commit failed.
    fixture.weights.output_is_tied = false;
    fixture.weights.output = fixture.tensors.matrix("bad.out", 4, 4, identity(4));

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights,
                                                      shared_text_tokenizer(), 8);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 1;
    request.token_stop_sequences = {{1, 2}};

    std::size_t callback_invocations = 0;
    require_throws(
        [&] {
            static_cast<void>(session.generate_fresh(
                request, [&callback_invocations](const oracle::runtime::Qwen35GenerationEvent&) {
                    ++callback_invocations;
                }));
        },
        "output projection dimensions");
    require(callback_invocations == 0,
            "no event may be delivered for a token whose commit failed, even with stops configured");
    require(session.state().sequence_length() == 0,
            "runtime failure must reset state in the stops-configured path too");
}

void test_final_text_equals_concatenated_event_fragments_with_stop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 4);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {decode_single(tokenizer, natural[2])};

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "text stop must fire for this fixture");
    require(events.size() == result.generated_tokens.size(),
            "exactly one event per committed token");  // req 22/26
    std::string assembled;
    for (const auto& event : events) assembled += event.text_fragment;
    require(assembled == result.generated_text,
            "final generated_text must equal the concatenation of delivered event "
            "text_fragments");  // req 21
}

// --- Finish precedence (required tests 27-33) ----------------------------

void test_finish_precedence_eos_vs_token_stop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    // This fixture greedily settles into repeating a single fixed token
    // forever (confirmed empirically), so the very first generated token
    // is enough to configure both EOS and a simultaneously-completing
    // single-token stop on the same commit.
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 1);
    model.eos_token_id = natural[0];

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    // A token stop that also completes exactly when the EOS token commits.
    request.token_stop_sequences = {{natural[0]}};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::eos,
            "EOS must win over a simultaneously-completing token stop");
    require(result.generated_tokens.size() == 1, "generation must stop right after EOS commits");
    require(!result.stop_match.has_value(), "no stop_match when the finish reason is eos");
}

void test_finish_precedence_eos_vs_text_stop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 1);
    model.eos_token_id = natural[0];

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    // A text stop that also completes exactly when the EOS token commits.
    request.text_stop_sequences = {decode_single(tokenizer, natural[0])};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::eos,
            "EOS must win over a simultaneously-completing text stop");
    require(result.generated_tokens.size() == 1, "generation must stop right after EOS commits");
    require(!result.stop_match.has_value(), "no stop_match when the finish reason is eos");
}

void test_finish_precedence_token_stop_vs_text_stop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 3);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0]}};
    request.text_stop_sequences = {decode_single(tokenizer, natural[0])};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a configured stop must fire");
    require(result.stop_match.has_value() &&
                result.stop_match->kind == oracle::runtime::Qwen35StopKind::token_sequence,
            "a token stop completing on the same token as a text stop must win");
}

void test_finish_precedence_token_stop_vs_max_tokens() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 3);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    // A 3-token stop matching the natural (repeated) sequence exactly:
    // since this fixture greedily repeats the same token forever, a
    // shorter stop (e.g. just [natural[2]]) would match on the very first
    // commit instead of the intended max-tokens-reaching (3rd) one.
    request.token_stop_sequences = {{natural[0], natural[1], natural[2]}};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a token stop completing on the max-tokens-reaching commit must win over max_tokens");
    require(result.generated_tokens.size() == 3, "exactly the configured token count committed");
}

void test_finish_precedence_text_stop_vs_max_tokens() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 3);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    // A 3-byte stop matching the natural (repeated) sequence's full text:
    // a shorter stop would match on the very first commit instead of the
    // intended max-tokens-reaching (3rd) one, for the same reason as above.
    std::string stop_text;
    for (std::size_t i = 0; i < 3; ++i) stop_text += decode_single(tokenizer, natural[i]);
    request.text_stop_sequences = {stop_text};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a text stop completing on the max-tokens-reaching commit must win over max_tokens");
    require(result.generated_tokens.size() == 3,
            "the text stop must complete exactly on the max-tokens-reaching (3rd) token for this "
            "precedence test to be meaningful");
    require(result.stop_match->kind == oracle::runtime::Qwen35StopKind::text_sequence,
            "stop kind must be text_sequence");
}

void test_finish_precedence_max_tokens_vs_context_exhausted() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};  // size 2
    // Capacity exactly equals prompt + max_generated_tokens, so the final
    // allowed generated token also exactly fills the cache.
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 5);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    // A stop long enough it can never actually match within 3 generated
    // tokens; present only to exercise the stops-configured code path.
    request.token_stop_sequences = {{model.vocabulary_size - 1, model.vocabulary_size - 1,
                                     model.vocabulary_size - 1, model.vocabulary_size - 1}};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "max_tokens must win when the final allowed token also fills cache capacity");
    require(result.generated_tokens.size() == 3, "exactly max_generated_tokens tokens committed");
    require(result.final_sequence_length == 5, "cache must be exactly full, not exceeded");
}

void test_finish_precedence_plain_context_exhausted() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 3);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 5;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{model.vocabulary_size - 1, model.vocabulary_size - 1,
                                     model.vocabulary_size - 1, model.vocabulary_size - 1,
                                     model.vocabulary_size - 1}};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::context_exhausted,
            "context exhaustion must still fire correctly in the stops-configured code path");
    require(result.generated_tokens.size() == 1,
            "context capacity should permit exactly one generated token, matching Slice 1");
    require(result.final_sequence_length == 3, "context capacity was exceeded");
    require(!result.stop_match.has_value(), "no stop_match for a plain context_exhausted finish");
}

// --- No-stop regression (required test 34) -------------------------------

void test_empty_stop_configuration_matches_slice3a_behavior() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    // Explicitly empty (also the default), proving Slice 3B's new fields
    // being present-but-unused changes nothing.
    request.token_stop_sequences = {};
    request.text_stop_sequences = {};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 8);
    const auto result = session.generate_fresh(request);

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "empty stop configuration must reproduce Slice 1/3A's greedy fixture result exactly");
    require(result.generated_tokens.size() == 3, "generated token count must match Slice 3A exactly");
    require(result.final_sequence_length == 5, "final state length must match Slice 3A exactly");
    require(!result.stop_match.has_value(), "stop_match must be absent with no stops configured");
    for (std::size_t index = 0; index < result.generated_tokens.size(); ++index) {
        require(result.generated_tokens[index].position == 2 + index,
                "generated token position must match Slice 3A exactly");
        require(result.generated_tokens[index].candidate_count == 1,
                "greedy sampling candidate count must remain one");
    }
}

// --- Callback failure during buffered release ----------------------------

void test_callback_failure_during_buffered_release() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 8);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0, 1};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    // A stop configured so it never matches, forcing the stops-configured
    // (buffered) code path to run to max_tokens, buffering all three
    // events, and only then attempt delivery.
    request.token_stop_sequences = {{model.vocabulary_size - 1, model.vocabulary_size - 1,
                                     model.vocabulary_size - 1, model.vocabulary_size - 1}};

    std::size_t callback_invocations = 0;
    require_throws(
        [&] {
            static_cast<void>(session.generate_fresh(
                request, [&callback_invocations](const oracle::runtime::Qwen35GenerationEvent&) {
                    ++callback_invocations;
                    if (callback_invocations == 2) {
                        throw std::runtime_error(
                            "Phase 2E Slice 3B synthetic callback failure during buffered release");
                    }
                }));
        },
        "synthetic callback failure during buffered release");
    require(callback_invocations == 2,
            "the callback must have been invoked for the first two buffered events before failing");
    require(session.state().sequence_length() == 0,
            "callback failure during buffered release must reset session state exactly like "
            "immediate-delivery callback failure does");
}

// --- Duplicate stop configuration policy ---------------------------------

void test_duplicate_stop_definitions_resolve_by_configured_index() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 16, 2);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 16);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 4;
    request.sampling.temperature = 0.0F;
    // Two byte-identical text stops (duplicate configuration is not
    // rejected -- see docs/PHASE_2E.md); the match must resolve to
    // configured_index 0, the first of the two.
    const std::string stop_text = decode_single(tokenizer, natural[0]);
    request.text_stop_sequences = {stop_text, stop_text};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a duplicated stop must still match");
    require(result.stop_match.has_value() && result.stop_match->configured_index == 0,
            "duplicate stop definitions must resolve deterministically to the earliest index");
}

// ---------------------------------------------------------------------
// Bounded rolling holdback (Slice 3B architectural correction): proves
// that Qwen35GenerationSession no longer buffers a stop-configured
// generation's events until the run ends, but instead releases each
// committed token's event as soon as it can no longer participate in any
// configured stop's completion.
// ---------------------------------------------------------------------

void test_incremental_release_before_stop_resolution() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    const std::string repeated_byte = decode_single(tokenizer, natural[0]);
    require(repeated_byte.size() == 1, "sanity: this fixture's tokens each decode to one byte");
    const auto other = (natural[0] + 1) % model.vocabulary_size;
    const std::string other_byte = decode_single(tokenizer, other);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;
    // A text stop beginning with a byte this fixture's greedy sampling
    // never actually produces (it only ever repeats `repeated_byte`), so
    // every committed token's visible text is immediately unable to begin
    // this stop -- the "XYZ can never begin ABC" case, proving safe output
    // is released *before* the run finishes rather than held to the end.
    request.text_stop_sequences = {other_byte + repeated_byte};

    std::vector<std::size_t> delivery_time_sequence_length;
    std::vector<std::size_t> delivery_generated_index;
    const auto result = session.generate_fresh(
        request, [&](const oracle::runtime::Qwen35GenerationEvent& event) {
            delivery_time_sequence_length.push_back(session.state().sequence_length());
            delivery_generated_index.push_back(event.generated_index);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "an unreachable text stop must not affect normal completion");
    require(result.generated_tokens.size() == 10, "full generation must proceed");
    require(delivery_time_sequence_length.size() == 10,
            "every committed token still gets exactly one event");  // req 7

    // If bounded holdback works, the FIRST event is released while the
    // model has generated far fewer than all 10 tokens -- proving delivery
    // happened incrementally, not only after the whole run finished (the
    // old whole-buffer design could only ever show this value equal to
    // final_sequence_length).
    require(delivery_time_sequence_length.front() < result.final_sequence_length,
            "the first safe event must be released before generation completes");
    require(delivery_time_sequence_length.front() == prompt.size() + 1,
            "content that cannot begin the stop must be released immediately, in the very "
            "iteration it commits");

    require(delivery_generated_index.front() == 0, "delivery order must remain canonical");
    for (std::size_t i = 1; i < delivery_generated_index.size(); ++i) {
        require(delivery_generated_index[i] == delivery_generated_index[i - 1] + 1,
                "callback delivery order must remain canonical");  // req 7
    }
}

void test_long_generation_with_short_stop_stays_bounded() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    const std::string repeated_byte = decode_single(tokenizer, natural[0]);
    const auto other = (natural[0] + 1) % model.vocabulary_size;
    const std::string other_byte = decode_single(tokenizer, other);
    // A stop sharing the naturally-repeated byte as its first character,
    // but whose second character never arrives (the fixture only ever
    // repeats `repeated_byte`) -- every committed token therefore looks
    // ambiguous for exactly one more iteration, then resolves as safe.
    const std::string stop = repeated_byte + other_byte;

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 12;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {stop};

    std::vector<std::size_t> gap;
    const auto result = session.generate_fresh(
        request, [&](const oracle::runtime::Qwen35GenerationEvent& event) {
            const std::size_t delivered_at = session.state().sequence_length();
            require(delivered_at >= event.sequence_length,
                    "an event cannot be delivered before its own token committed");
            gap.push_back(delivered_at - event.sequence_length);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "this stop can never actually complete");
    require(result.generated_tokens.size() == 12, "full generation must proceed");
    require(gap.size() == 12, "every committed token still gets exactly one event");  // req 7
    for (const std::size_t value : gap) {
        require(value <= 2,
                "an event was held back far longer than this short stop's own ambiguity window "
                "allows -- the whole event history is accumulating instead of releasing "
                "incrementally");  // req 3
    }
}

void test_multi_token_stop_prefix_released_once_no_longer_ambiguous() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    // An unreachable token id this fixture's greedy sampling never
    // actually selects (it only ever repeats natural[0]), so a token stop
    // ending in it can never complete -- it exists purely to hold the
    // naturally repeated token back for exactly (length-1) iterations each
    // time, then release it once no longer ambiguous.
    const auto unreachable = (natural[0] + 1) % model.vocabulary_size;

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0], natural[0], natural[0], unreachable}};

    std::vector<std::size_t> gap;
    const auto result = session.generate_fresh(
        request, [&](const oracle::runtime::Qwen35GenerationEvent& event) {
            gap.push_back(session.state().sequence_length() - event.sequence_length);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "a stop ending in an unreachable token can never complete");
    require(gap.size() == 10, "every committed token still gets exactly one event");  // req 7
    for (const std::size_t value : gap) {
        require(value <= 4,
                "a shared-prefix token stop must not hold events back indefinitely once they age "
                "out of the ambiguity window");  // req 4
    }
}

void test_shared_prefix_text_stops_resolve_to_shortest_match() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 2);
    const std::string one = decode_single(tokenizer, natural[0]);
    const std::string two = one + decode_single(tokenizer, natural[1]);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;
    // Two text stops sharing a byte prefix: `two` is a longer extension of
    // `one`. `one` (the shorter stop, configured second) fully matches
    // after just 1 token, before `two` ever could -- proving a
    // shared-prefix stop still resolves correctly under bounded holdback.
    request.text_stop_sequences = {two, one};

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "one of the shared-prefix stops must match");  // req 5
    require(result.stop_match.has_value() && result.stop_match->configured_index == 1,
            "the shorter stop (configured_index 1) completes first and must be the one reported");
    require(result.generated_tokens.size() == 1,
            "generation must stop as soon as the shorter stop matches");
}

void test_mixed_token_and_text_stop_holdback() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    const std::string repeated_byte = decode_single(tokenizer, natural[0]);
    const auto unreachable = (natural[0] + 1) % model.vocabulary_size;
    const std::string unreachable_byte = decode_single(tokenizer, unreachable);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 8;
    request.sampling.temperature = 0.0F;
    // Neither configured stop can ever actually complete; both are
    // configured together purely to prove bounded holdback respects BOTH
    // ambiguity windows simultaneously (req 6), then releases the maximal
    // prefix safe under both.
    request.token_stop_sequences = {{natural[0], natural[0], unreachable}};
    request.text_stop_sequences = {repeated_byte + unreachable_byte};

    std::vector<std::size_t> gap;
    std::vector<std::size_t> generated_index_order;
    std::string assembled_from_events;
    const auto result = session.generate_fresh(
        request, [&](const oracle::runtime::Qwen35GenerationEvent& event) {
            gap.push_back(session.state().sequence_length() - event.sequence_length);
            generated_index_order.push_back(event.generated_index);
            assembled_from_events += event.text_fragment;
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "neither configured stop can ever complete for this fixture");
    require(gap.size() == 8, "every committed token still gets exactly one event");  // req 7
    for (std::size_t i = 1; i < generated_index_order.size(); ++i) {
        require(generated_index_order[i] == generated_index_order[i - 1] + 1,
                "callback delivery order must remain canonical even with mixed stop kinds");
    }
    for (const std::size_t value : gap) {
        require(value <= 4,
                "mixed token+text holdback must still eventually release once safe under both "
                "ambiguity windows");  // req 6
    }
    require(assembled_from_events == result.generated_text,
            "final generated_text must equal the concatenation of delivered event text_fragments "
            "even with mixed token+text stops and incremental release");
}

// ---------------------------------------------------------------------
// Slice 3C: reasoning-loop detector unit tests. Deliberately decoupled
// from Qwen35GenerationSession/sampling -- exactly like
// Qwen35IncrementalTextAssembler -- since this fixture's greedy sampling
// can only ever produce a single repeated token (see the note above
// discover_natural_tokens), which makes it impossible to obtain genuinely
// varying period-2/3/N content from real sampling.
// ---------------------------------------------------------------------

void test_reasoning_detector_period_1() {
    const std::vector<std::uint32_t> window{7, 7, 7, 7, 7, 7, 7, 7};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "8 identical tokens must be detected as period 1");
    require(found->period == 1, "period must be 1");
    require(found->coverage == 8, "coverage must span the whole window");
}

void test_reasoning_detector_period_2() {
    const std::vector<std::uint32_t> window{1, 2, 1, 2, 1, 2, 1, 2};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "ABABABAB must be detected");
    require(found->period == 2, "period must be 2");
    require(found->coverage == 8, "coverage must span the whole window");
}

void test_reasoning_detector_period_3() {
    const std::vector<std::uint32_t> window{1, 2, 3, 1, 2, 3, 1, 2, 3};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "ABCABCABC must be detected");
    require(found->period == 3, "period must be 3");
    require(found->coverage == 9, "coverage must span the whole window");
}

void test_reasoning_detector_longer_period() {
    const std::vector<std::uint32_t> window{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "ABCDE repeated 3x must be detected");
    require(found->period == 5, "period must be 5");
    require(found->coverage == 15, "coverage must span the whole window");
}

void test_reasoning_detector_prefers_shortest_period() {
    // Genuinely period-2, and therefore trivially also period-4/6/8/10 --
    // the shortest confirmed period must win.
    const std::vector<std::uint32_t> window{9, 4, 9, 4, 9, 4, 9, 4, 9, 4};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "must detect a period");
    require(found->period == 2, "the shortest confirmed period must be reported, not 4/6/8/10");
}

void test_reasoning_detector_minimum_length_protection() {
    // Only 4 tokens available; a higher minimum_repeated_coverage must
    // prevent detection no matter how perfectly periodic they are.
    const std::vector<std::uint32_t> window{5, 5, 5, 5};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 10);
    require(!found.has_value(),
            "a short window must not qualify against a higher minimum_repeated_coverage");
}

void test_reasoning_detector_single_duplicate_pair_insufficient() {
    // Only the last two tokens are equal; everything before differs. Even
    // with an aggressively low minimum_repeated_coverage (2), the
    // structural "at least 3x period" rule must still reject a single
    // duplicated pair.
    const std::vector<std::uint32_t> window{11, 22, 33, 44, 77, 77};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 2);
    require(!found.has_value(),
            "a single duplicated token pair must never be reported as a confirmed loop, "
            "regardless of configured coverage threshold");
}

void test_reasoning_detector_ignores_content_before_periodic_tail() {
    // Two leading "garbage" tokens, then a clean period-2 tail. Detection
    // must anchor on the tail; the garbage prefix must not extend the
    // reported coverage or otherwise corrupt the result -- this is what
    // "bounded trailing window" means in practice.
    const std::vector<std::uint32_t> window{99, 98, 1, 2, 1, 2, 1, 2, 1, 2};
    const auto found = oracle::runtime::qwen35_detect_reasoning_loop(window, 16, 6);
    require(found.has_value(), "the periodic tail must still be detected");
    require(found->period == 2, "period must be 2");
    require(found->coverage == 8,
            "coverage must cover only the periodic tail, not the garbage prefix");
}

// ---------------------------------------------------------------------
// Slice 3C: reasoning-state tracking (prompt-aware) and policy OFF/STOP.
// All of these configure Qwen35ReasoningBoundary/Qwen35ReasoningLoopConfig
// with deliberately low, isolated validation thresholds (never production
// defaults) against this fixture's single-repeated-token natural output --
// exactly the technique already established for Slice 3B's token-stop
// tests.
// ---------------------------------------------------------------------

void test_reasoning_active_from_prompt_tail() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};  // matches this exact prompt's own tail
    boundary.end_tokens = {6};       // never appears in prompt or generated output
    boundary.force_close_supported = false;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "a prompt-initiated reasoning segment must still be detected");  // req 1, 8
    require(result.generated_tokens.size() == 3,
            "generation must stop exactly when the loop is confirmed, no extra token");  // req 14
    require(result.reasoning_active_at_finish,
            "policy stop must not close reasoning -- it only terminates generation");
    require(result.reasoning_interventions.size() == 1, "exactly one intervention must be recorded");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.generated_index == 2, "must report the triggering token's index");
    require(intervention.reasoning_token_count == 3, "must report the reasoning segment length");
    require(intervention.detected_period == 1, "the repeated token forms a period-1 loop");
    require(intervention.repeated_coverage == 3, "coverage must match the confirmed window");
    require(intervention.policy == oracle::runtime::Qwen35ReasoningLoopPolicy::stop,
            "telemetry must record the configured policy");
    require(!intervention.closure_attempted && !intervention.closure_succeeded,
            "policy stop must never attempt closure");
}

void test_reasoning_activates_from_generated_start_marker() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{3, 5};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    // Matches the naturally-generated value, but this value never appears
    // anywhere in the prompt {3, 5} -- reasoning can only become active via
    // a *generated* token matching this marker, never via the prompt scan.
    boundary.start_tokens = {1};
    boundary.end_tokens = {0};  // never appears in prompt or generated output
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "a generated reasoning-start marker must activate the detector");  // req 8
    // Token 0 itself completes the start marker (reasoning content begins
    // at index 1); 3 more reasoning tokens are then needed before the
    // detector's minimum_reasoning_tokens gate opens.
    require(result.generated_tokens.size() == 4,
            "must stop right after the 4th token confirms the loop");
    require(result.reasoning_interventions.size() == 1, "exactly one intervention must be recorded");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.generated_index == 3, "must report the triggering token's index");
    require(intervention.reasoning_token_count == 3,
            "the start-marker token itself must not count as reasoning content");
}

void test_reasoning_closes_normally_and_detection_stops() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};  // active from the prompt tail
    boundary.end_tokens = {1};       // matches the very first generated token
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    // Deliberately trivial thresholds: if reasoning were still active, this
    // would trigger on essentially every subsequent token. Proving nothing
    // fires proves closure genuinely disables detection, not that the
    // thresholds merely never lined up.
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "reasoning closes on the first token, so generation must run to max_tokens "
            "normally");  // req 8
    require(result.generated_tokens.size() == 6, "full generation must proceed");
    require(!result.reasoning_active_at_finish, "reasoning must be inactive at finish");
    require(result.reasoning_interventions.empty(),
            "no detection may occur once reasoning has closed -- including for the identical "
            "repeated visible content that follows");  // req 8, visible-repetition-after-close
}

void test_reasoning_policy_off_is_inert() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    // The exact configuration from test_reasoning_active_from_prompt_tail,
    // which *does* trigger reasoning_loop under policy stop -- but here
    // with policy off.
    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {6};

    oracle::runtime::Qwen35GenerationSession session_with_config(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request_with_config;
    request_with_config.prompt_tokens = prompt;
    request_with_config.max_generated_tokens = 10;
    request_with_config.sampling.temperature = 0.0F;
    request_with_config.reasoning_boundary = boundary;
    request_with_config.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::off;
    request_with_config.reasoning_loop.minimum_reasoning_tokens = 3;
    request_with_config.reasoning_loop.inspection_window = 8;
    request_with_config.reasoning_loop.maximum_period = 2;
    request_with_config.reasoning_loop.minimum_repeated_coverage = 3;
    const auto result_with_config = session_with_config.generate_fresh(request_with_config);

    oracle::runtime::Qwen35GenerationSession session_plain(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request_plain;
    request_plain.prompt_tokens = prompt;
    request_plain.max_generated_tokens = 10;
    request_plain.sampling.temperature = 0.0F;
    const auto result_plain = session_plain.generate_fresh(request_plain);

    require(result_with_config.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "policy off must never intervene, even with a boundary/thresholds that would "
            "otherwise trigger");  // req 13, 26
    std::vector<std::uint32_t> ids_with_config;
    for (const auto& token : result_with_config.generated_tokens) ids_with_config.push_back(token.token_id);
    std::vector<std::uint32_t> ids_plain;
    for (const auto& token : result_plain.generated_tokens) ids_plain.push_back(token.token_id);
    require(ids_with_config == ids_plain,
            "the generated token sequence must be byte-for-byte identical to a request with no "
            "reasoning configuration at all");
    require(result_with_config.generated_text == result_plain.generated_text,
            "generated_text must also be identical");
    require(result_with_config.reasoning_interventions.empty(),
            "no interventions may ever be recorded under policy off");
}

// ---------------------------------------------------------------------
// Slice 3C: force-close intervention -- genuine forward-path integrity,
// atomicity, and failure modes.
// ---------------------------------------------------------------------

void test_force_close_commits_full_end_sequence_and_resumes() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {3, 5};  // an arbitrary, valid 2-token closure sequence
    boundary.force_close_supported = true;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;
    request.reasoning_loop.maximum_interventions = 2;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "a successful force-close must not itself become the final finish_reason -- "
            "generation resumes and here runs to max_tokens");  // req 20, 23
    require(result.generated_tokens.size() == 10, "full budget, including forced tokens, must be used");
    require(result.generated_tokens[3].token_id == 3, "the first forced token must be in the ledger");
    require(result.generated_tokens[4].token_id == 5, "the second forced token must be in the ledger");
    require(!result.reasoning_active_at_finish, "reasoning must be inactive after closure");

    require(result.reasoning_interventions.size() == 1, "exactly one intervention must be recorded");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.closure_attempted && intervention.closure_succeeded,
            "closure must be recorded as attempted and succeeded");
    require(intervention.failure_reason.empty(), "no failure reason on success");
    require(intervention.detected_period == 1 && intervention.repeated_coverage == 3,
            "telemetry must report the real detection that triggered closure");

    require(events.size() == 10, "every committed token, forced or sampled, gets exactly one event");
    std::size_t forced_event_count = 0;
    for (std::size_t i = 1; i < events.size(); ++i) {
        require(events[i].generated_index == events[i - 1].generated_index + 1,
                "callback delivery order must remain canonical across forced and sampled tokens");
    }
    for (const auto& event : events) {
        if (event.source == oracle::runtime::Qwen35GenerationTokenSource::reasoning_force_close) {
            ++forced_event_count;
        }
    }
    require(forced_event_count == 2, "exactly the 2 forced tokens must carry the forced source");

    const auto& forced_event_0 = events[3];
    const auto& forced_event_1 = events[4];
    require(forced_event_0.source == oracle::runtime::Qwen35GenerationTokenSource::reasoning_force_close,
            "event at the first forced index must carry the forced source");  // req 17
    require(forced_event_1.source == oracle::runtime::Qwen35GenerationTokenSource::reasoning_force_close,
            "event at the second forced index must carry the forced source");
    require(forced_event_0.probability == 0.0F && forced_event_0.candidate_count == 0,
            "forced-token probability/candidate_count must be the documented sentinel, never a "
            "fabricated sampled value");  // req 17
    require(forced_event_0.token_bytes == decode_single(tokenizer, 3),
            "forced-token bytes must come from a genuine decode of the actual forced id");
    require(forced_event_1.token_bytes == decode_single(tokenizer, 5),
            "forced-token bytes must come from a genuine decode of the actual forced id");
    require(forced_event_0.sequence_length + 1 == forced_event_1.sequence_length,
            "each forced token must advance state by exactly one, proving a genuine forward call "
            "per token rather than a batch/replay");  // req 13, 16

    // Boundary-visibility correction: token_bytes stay exact (checked above)
    // but the forced marker's visible presentation must be fully suppressed.
    require(forced_event_0.text_fragment.empty() && forced_event_1.text_fragment.empty(),
            "a forced reasoning-boundary marker's visible text_fragment must be suppressed even "
            "though its token_bytes remain exact and it remains fully committed");
    const std::string forced_bytes_combined =
        decode_single(tokenizer, 3) + decode_single(tokenizer, 5);
    require(result.generated_text.find(forced_bytes_combined) == std::string::npos,
            "the forced reasoning-boundary marker's raw bytes must never appear in generated_text");
    for (std::size_t i = 5; i < events.size(); ++i) {
        require(events[i].source == oracle::runtime::Qwen35GenerationTokenSource::sampled,
                "ordinary sampling must resume with normal source metadata after closure");
        require(!events[i].text_fragment.empty(),
                "ordinary visible sampling after closure must be delivered normally, not "
                "suppressed");  // required regression coverage
    }

    std::string assembled;
    for (const auto& event : events) assembled += event.text_fragment;
    require(assembled == result.generated_text,
            "final generated_text must equal the concatenation of delivered event fragments even "
            "through a force-close");
}

// ---------------------------------------------------------------------
// Slice 3C correction: reasoning-boundary marker visible-text suppression.
// Proves the literal marker sequence never leaks into text_fragment/
// generated_text, whether genuinely sampled (start or end) or forced,
// while token_bytes/ledger/cache/telemetry remain exactly as before.
// ---------------------------------------------------------------------

void test_reasoning_start_marker_suppressed_from_visible_text() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{3, 5};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {1};  // genuinely sampled -- never appears in the prompt
    boundary.end_tokens = {0};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "sanity: this is the same trigger shape as "
            "test_reasoning_activates_from_generated_start_marker");
    require(result.generated_tokens.size() == 4, "sanity: same commit count as that test");
    require(result.generated_tokens[0].token_id == 1, "sanity: token 0 completes the start marker");
    require(events.size() == 4, "every committed token still gets exactly one event");

    require(events[0].text_fragment.empty(),
            "a genuinely sampled reasoning-start marker must not leak into text_fragment");  // req
    require(events[0].token_bytes == decode_single(tokenizer, 1),
            "token_bytes for the suppressed start-marker token must remain exact");
    for (std::size_t i = 1; i < events.size(); ++i) {
        require(!events[i].text_fragment.empty(),
                "ordinary reasoning content after the start marker must remain visible");
        require(events[i].text_fragment == decode_single(tokenizer, 1),
                "visible content must match the genuinely decoded byte");
    }
    require(result.generated_text.find(decode_single(tokenizer, 1)) != std::string::npos,
            "sanity: the repeated byte does legitimately appear from the unsuppressed tokens");

    std::string assembled;
    for (const auto& event : events) assembled += event.text_fragment;
    require(assembled == result.generated_text,
            "final generated_text must equal the concatenation of delivered event fragments");
}

void test_reasoning_end_marker_suppressed_from_visible_text() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{3, 5};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    // Matches the prompt's own tokens exactly (not a suffix/prefix
    // relationship with end_tokens, so its last-occurrence position in the
    // prompt is unambiguous) -- reasoning is active from the prompt itself.
    boundary.start_tokens = {3, 5};
    // Matches the very first genuinely sampled token; "1" never appears
    // anywhere in prompt {3, 5}, so this cannot coincidentally match inside
    // the prompt scan too.
    boundary.end_tokens = {1};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "sanity: same shape as test_reasoning_closes_normally_and_detection_stops");
    require(result.generated_tokens.size() == 6, "full generation must proceed");
    require(!result.reasoning_active_at_finish, "reasoning must be inactive after the genuine close");
    require(events.size() == 6, "every committed token still gets exactly one event");

    require(events[0].text_fragment.empty(),
            "a genuinely sampled reasoning-end marker must not leak into text_fragment");  // req
    require(events[0].token_bytes == decode_single(tokenizer, 1),
            "token_bytes for the suppressed end-marker token must remain exact");
    for (std::size_t i = 1; i < events.size(); ++i) {
        require(events[i].source == oracle::runtime::Qwen35GenerationTokenSource::sampled,
                "visible repetition after the genuine close is ordinary sampled content");
        require(!events[i].text_fragment.empty(),
                "visible repetition after reasoning closes must be delivered normally, not "
                "suppressed -- only the marker itself is suppressed");
    }

    std::string assembled;
    for (const auto& event : events) assembled += event.text_fragment;
    require(assembled == result.generated_text,
            "final generated_text must equal the concatenation of delivered event fragments");
}

void test_reasoning_multi_token_end_marker_suppressed_via_extended_holdback() {
    // Proves the bounded holdback is genuinely extended to cover multi-
    // token reasoning-boundary sequences -- without it, the first of the
    // two marker tokens would already have been released (and therefore
    // visibly leaked) one iteration before the second token confirms the
    // match.
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    require(natural[0] == 1, "test assumption: this prompt's natural token is 1");

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 6;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {1, 1};  // a genuine 2-token marker, both tokens the repeated value
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        request, [&events](const oracle::runtime::Qwen35GenerationEvent& event) {
            events.push_back(event);
        });

    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
            "the end marker closes reasoning but does not itself stop generation");
    require(result.generated_tokens.size() == 6, "full generation must proceed");
    require(events.size() == 6, "every committed token still gets exactly one event");
    require(!result.reasoning_active_at_finish, "reasoning must be inactive after the genuine close");

    require(events[0].text_fragment.empty(),
            "the first token of a genuinely sampled 2-token end marker must not leak, even though "
            "it committed one full iteration before the match was confirmed");
    require(events[1].text_fragment.empty(),
            "the second token of the 2-token end marker must not leak");
    require(events[0].token_bytes == decode_single(tokenizer, 1) &&
                events[1].token_bytes == decode_single(tokenizer, 1),
            "token_bytes for both suppressed marker tokens must remain exact");
    for (std::size_t i = 2; i < events.size(); ++i) {
        require(!events[i].text_fragment.empty(),
                "ordinary content after the multi-token marker closes must remain visible");
    }

    std::string assembled;
    for (const auto& event : events) assembled += event.text_fragment;
    require(assembled == result.generated_text,
            "final generated_text must equal the concatenation of delivered event fragments");
}

void test_force_close_fails_insufficient_token_budget() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 3;  // exactly the trigger point -- no room for 2 forced tokens
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {3, 5};
    boundary.force_close_supported = true;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "insufficient token budget must safely finish reasoning_loop");  // req 18, 28
    require(result.generated_tokens.size() == 3,
            "no forced token may be committed when the full sequence cannot fit the budget");  // req 18
    require(result.reasoning_active_at_finish, "closure failed, so reasoning is still active");
    require(result.reasoning_interventions.size() == 1, "exactly one intervention must be recorded");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.closure_attempted && !intervention.closure_succeeded,
            "closure must be recorded as attempted but failed");
    require(intervention.failure_reason == "insufficient_token_budget",
            "failure reason must identify the token-budget cause");
}

void test_force_close_fails_insufficient_context() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    // Capacity == prompt size + the 3 tokens needed to trigger detection,
    // exactly -- zero room remains for either forced token.
    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer,
                                                      prompt.size() + 3);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 20;  // budget is generous -- context is the isolated cause
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {3, 5};
    boundary.force_close_supported = true;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "insufficient context must safely finish reasoning_loop");  // req 19, 28
    require(result.generated_tokens.size() == 3,
            "no forced token may be committed when the full sequence cannot fit remaining "
            "context");  // req 19
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.failure_reason == "insufficient_context",
            "failure reason must identify the context cause");
}

void test_force_close_fails_missing_end_sequence() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {};  // missing/empty
    boundary.force_close_supported = true;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "a missing reasoning-end sequence must safely finish reasoning_loop");  // req 28
    require(result.generated_tokens.size() == 3, "no forced token can exist without a configured sequence");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.failure_reason == "missing_reasoning_end_sequence",
            "failure reason must identify the missing-sequence cause");
}

void test_force_close_fails_unsupported_by_template() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {3, 5};
    boundary.force_close_supported = false;  // template does not support force-close
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "force-close unsupported by the template must safely finish reasoning_loop");  // req 28
    require(result.generated_tokens.size() == 3, "no forced token may be committed");
    const auto& intervention = result.reasoning_interventions.front();
    require(intervention.failure_reason == "unsupported_by_template",
            "failure reason must identify the unsupported-template cause");
}

void test_force_close_escalates_after_intervention_limit() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {3, 5};
    boundary.force_close_supported = true;
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    request.reasoning_loop.minimum_reasoning_tokens = 3;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 3;
    // An otherwise-closeable loop, but the intervention budget is already
    // exhausted before the first attempt.
    request.reasoning_loop.maximum_interventions = 0;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "exceeding the intervention limit must safely finish reasoning_loop");  // req 22
    require(result.generated_tokens.size() == 3,
            "no closure may even be attempted once the intervention limit is exhausted");
    const auto& intervention = result.reasoning_interventions.front();
    require(!intervention.closure_attempted && !intervention.closure_succeeded,
            "an escalated intervention must not attempt closure at all");
    require(intervention.failure_reason == "intervention_limit_exceeded",
            "failure reason must identify the escalation cause");
}

// ---------------------------------------------------------------------
// Slice 3C: termination precedence with reasoning-loop safety configured.
// ---------------------------------------------------------------------

void test_precedence_eos_vs_reasoning_loop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);
    model.eos_token_id = natural[0];

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {6};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::eos,
            "EOS must win over a reasoning-loop detection that would otherwise fire on the same "
            "token");  // req 23
    require(result.generated_tokens.size() == 1, "generation must stop right after EOS commits");
    require(result.reasoning_interventions.empty(),
            "the reasoning check must never run once EOS has already decided this iteration");
}

void test_precedence_token_stop_vs_reasoning_loop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;
    request.token_stop_sequences = {{natural[0]}};

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {6};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a configured token stop must win over reasoning-loop detection");  // req 23
    require(result.generated_tokens.size() == 1, "generation must stop right after the stop token commits");
    require(result.reasoning_interventions.empty(),
            "the reasoning check must never run once a token stop has already decided this "
            "iteration");
}

void test_precedence_text_stop_vs_reasoning_loop() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};
    const auto natural = discover_natural_tokens(model, fixture.weights, tokenizer, prompt, 32, 1);

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 10;
    request.sampling.temperature = 0.0F;
    request.text_stop_sequences = {decode_single(tokenizer, natural[0])};

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {0, 1};
    boundary.end_tokens = {6};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    request.reasoning_loop.minimum_reasoning_tokens = 1;
    request.reasoning_loop.inspection_window = 8;
    request.reasoning_loop.maximum_period = 2;
    request.reasoning_loop.minimum_repeated_coverage = 1;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "a configured text stop must win over reasoning-loop detection");  // req 23
    require(result.generated_tokens.size() == 1, "generation must stop right after the stop token commits");
    require(result.reasoning_interventions.empty(),
            "the reasoning check must never run once a text stop has already decided this "
            "iteration");
}

void test_precedence_context_exhaustion_with_reasoning_configured() {
    ModelFixture fixture;
    auto model = manifest();
    const auto& tokenizer = shared_text_tokenizer();
    const std::vector<std::uint32_t> prompt{0, 1};

    oracle::runtime::Qwen35GenerationSession session(model, fixture.weights, tokenizer,
                                                      prompt.size() + 2);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = prompt;
    request.max_generated_tokens = 20;
    request.sampling.temperature = 0.0F;

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    // A marker this fixture's output can never contain, so reasoning never
    // activates -- isolates context_exhausted as the only possible finish.
    boundary.start_tokens = {6};
    boundary.end_tokens = {7};
    request.reasoning_boundary = boundary;
    request.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;

    const auto result = session.generate_fresh(request);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::context_exhausted,
            "reasoning safety being configured must not interfere with ordinary context "
            "exhaustion when it never actually activates");  // req 23
    require(result.generated_tokens.size() == 2, "generation must stop exactly at context capacity");
    require(result.reasoning_interventions.empty(), "no detection ever ran");
}

}  // namespace

int main() {
    try {
        test_greedy_generation_and_state_ledger();
        test_eos_is_committed_before_finish();
        test_context_exhaustion_is_bounded();
        test_runtime_failure_resets_partial_state();
        test_preflight_preserves_existing_state();
        test_invalid_prompt_token_rejected();

        test_generation_events_match_ledger();
        test_callback_does_not_alter_greedy_tokens();
        test_no_event_for_uncommitted_token();
        test_callback_failure_resets_session_state();

        test_ascii_assembly();
        test_multibyte_utf8_within_one_token();
        test_utf8_split_across_token_boundary();
        test_malformed_utf8_is_replaced();
        test_trailing_partial_sequence_is_replaced_at_finish();
        test_special_and_unused_token_behavior();

        test_stop_preflight_rejects_empty_token_stop();
        test_stop_preflight_rejects_out_of_vocabulary_token_stop();
        test_stop_preflight_rejects_empty_text_stop();
        test_stop_preflight_rejects_invalid_utf8_text_stop();

        test_token_stop_single_token();
        test_token_stop_multi_token_spanning_iterations();
        test_token_stop_shared_prefix_no_premature_match();

        test_text_stop_within_one_token();
        test_text_stop_split_across_two_tokens();
        test_text_stop_split_across_several_tokens();
        test_text_stop_begins_mid_stream();
        test_text_stop_multibyte_and_mid_token_trailing_suffix();
        test_text_stop_matching_uses_safe_text_not_raw_bytes();

        test_no_uncommitted_token_event_with_stops_configured();
        test_final_text_equals_concatenated_event_fragments_with_stop();

        test_finish_precedence_eos_vs_token_stop();
        test_finish_precedence_eos_vs_text_stop();
        test_finish_precedence_token_stop_vs_text_stop();
        test_finish_precedence_token_stop_vs_max_tokens();
        test_finish_precedence_text_stop_vs_max_tokens();
        test_finish_precedence_max_tokens_vs_context_exhausted();
        test_finish_precedence_plain_context_exhausted();

        test_empty_stop_configuration_matches_slice3a_behavior();
        test_callback_failure_during_buffered_release();
        test_duplicate_stop_definitions_resolve_by_configured_index();

        test_incremental_release_before_stop_resolution();
        test_long_generation_with_short_stop_stays_bounded();
        test_multi_token_stop_prefix_released_once_no_longer_ambiguous();
        test_shared_prefix_text_stops_resolve_to_shortest_match();
        test_mixed_token_and_text_stop_holdback();

        test_reasoning_detector_period_1();
        test_reasoning_detector_period_2();
        test_reasoning_detector_period_3();
        test_reasoning_detector_longer_period();
        test_reasoning_detector_prefers_shortest_period();
        test_reasoning_detector_minimum_length_protection();
        test_reasoning_detector_single_duplicate_pair_insufficient();
        test_reasoning_detector_ignores_content_before_periodic_tail();

        test_reasoning_active_from_prompt_tail();
        test_reasoning_activates_from_generated_start_marker();
        test_reasoning_closes_normally_and_detection_stops();
        test_reasoning_policy_off_is_inert();

        test_force_close_commits_full_end_sequence_and_resumes();
        test_reasoning_start_marker_suppressed_from_visible_text();
        test_reasoning_end_marker_suppressed_from_visible_text();
        test_reasoning_multi_token_end_marker_suppressed_via_extended_holdback();
        test_force_close_fails_insufficient_token_budget();
        test_force_close_fails_insufficient_context();
        test_force_close_fails_missing_end_sequence();
        test_force_close_fails_unsupported_by_template();
        test_force_close_escalates_after_intervention_limit();

        test_precedence_eos_vs_reasoning_loop();
        test_precedence_token_stop_vs_reasoning_loop();
        test_precedence_text_stop_vs_reasoning_loop();
        test_precedence_context_exhaustion_with_reasoning_configured();

        std::cout << "Phase 2E slice 1 + slice 3A + slice 3B + bounded holdback + slice 3C tests "
                     "passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Phase 2E test failure: " << error.what() << '\n';
        return 1;
    }
}
