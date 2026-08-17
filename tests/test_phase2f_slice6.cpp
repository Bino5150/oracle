// Phase 2F Slice 6: guarded MTP generation-session integration
// (oracle/runtime/qwen35_generation.hpp's Qwen35MtpGenerationConfig/
// Qwen35MtpDiagnostics, and the obtain_next_token() lambda inside
// Qwen35GenerationSession::generate_fresh). Hand-controlled synthetic tests
// that do not require the real checkpoint; the real 2B MTP ON/OFF fixture is
// evidence-only, recorded in
// model-reports/phase2f-slice1-mtp-baseline-20260816/slice6-generation/ (not
// committed).
//
// Design note (see docs/PHASE_2F.md, "Slice 6"): every committed token, MTP
// ON or OFF, passes through the exact same real forward + real greedy
// sampler call. MTP only ever *predicts* a token in advance (via Slice 2-5's
// already-validated draft/verify machinery on a disposable shadow copy of
// canonical state) and asserts that the real, independently-derived sample
// agrees -- it can never itself change what gets committed. This means
// EVERY test below that compares "MTP OFF" against "MTP ON" is expected to
// find them identical by construction; what these tests actually exercise
// is (a) that the integration code doesn't misbehave -- crash, double-commit,
// leak a rejected/unused draft as a canonical event, get depth-bounding or
// termination precedence wrong -- when a real MTP decision batch spans an
// interesting boundary (EOS, stop, UTF-8 split, reasoning marker, max_tokens,
// context capacity), and (b) that the diagnostic accounting is correct.

#include "oracle/runtime/qwen35_generation.hpp"
#include "oracle/runtime/qwen35_state_fingerprint.hpp"
#include "oracle/model/ggml_type.hpp"
#include "oracle/model/gguf.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
                "exception message did not contain expected text: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

// --- shared fixture-building scaffolding (matches the pattern established
// in tests/test_phase2f_slice4.cpp / test_phase2f_slice5.cpp) -------------

oracle::model::Qwen35Manifest manifest(std::uint32_t vocabulary_size, std::uint32_t embedding_length,
                                       std::uint32_t block_count = 2, bool with_mtp = true) {
    oracle::model::Qwen35Manifest value;
    value.architecture = "qwen35";
    value.model_name = "phase2f-slice6-fixture";
    value.backbone_block_count = block_count;
    value.nextn_predict_layers = with_mtp ? 1U : 0U;
    value.total_block_count = value.backbone_block_count + value.nextn_predict_layers;
    value.context_length = 1024;
    value.embedding_length = embedding_length;
    value.feed_forward_length = 32;
    value.attention_head_count = 2;
    value.attention_head_count_kv = 1;
    value.attention_key_length = 8;
    value.attention_value_length = 8;
    value.attention_rms_epsilon = 1.0e-6F;
    value.rope_dimension_count = 8;
    value.rope_dimension_sections = {2, 1, 1, 0};
    value.rope_frequency_base = 10000.0F;
    value.ssm_convolution_kernel = 4;
    value.ssm_state_size = 4;
    value.ssm_group_count = 1;
    value.ssm_time_step_rank = 2;
    value.ssm_inner_size = 8;
    value.full_attention_interval = 2;
    value.vocabulary_size = vocabulary_size;
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

    void add(std::string name, std::initializer_list<std::uint64_t> dimensions,
             std::uint32_t type = 0) {
        infos.push_back({std::move(name), dimensions, type, 0});
        const auto* layout = oracle::model::ggml_type_layout(type);
        require(layout != nullptr, "fixture type layout missing");
        const std::uint64_t bytes = oracle::model::ggml_tensor_byte_size(infos.back().dimensions, type);
        storage.emplace_back(static_cast<std::size_t>(bytes));
        views.emplace_back(&infos.back(), layout, storage.back().data(), storage.back().size(), 0);
    }

    void set_f32(const std::string& name, const std::vector<float>& values) {
        for (std::size_t index = 0; index < infos.size(); ++index) {
            if (infos[index].name != name) continue;
            require(storage[index].size() == values.size() * sizeof(float),
                    "fixture set_f32 size mismatch for " + name);
            std::memcpy(storage[index].data(), values.data(), storage[index].size());
            return;
        }
        throw std::runtime_error("fixture set_f32: tensor not found: " + name);
    }
};

void add_attention(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    const std::uint64_t q_width = model.attention_key_length * model.attention_head_count * 2U;
    const std::uint64_t k_width = model.attention_key_length * model.attention_head_count_kv;
    const std::uint64_t v_width = model.attention_value_length * model.attention_head_count_kv;
    const std::uint64_t out_width = model.attention_key_length * model.attention_head_count;
    fixture.add(prefix + "attn_q.weight", {model.embedding_length, q_width});
    fixture.add(prefix + "attn_k.weight", {model.embedding_length, k_width});
    fixture.add(prefix + "attn_v.weight", {model.embedding_length, v_width});
    fixture.add(prefix + "attn_output.weight", {out_width, model.embedding_length});
    fixture.add(prefix + "attn_q_norm.weight", {model.attention_key_length});
    fixture.add(prefix + "attn_k_norm.weight", {model.attention_key_length});
}

void add_recurrent(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    const std::uint64_t key_width = model.ssm_state_size * model.ssm_group_count;
    const std::uint64_t value_width = model.ssm_state_size * model.ssm_time_step_rank;
    const std::uint64_t conv_width = key_width * 2U + value_width;
    fixture.add(prefix + "attn_qkv.weight", {model.embedding_length, conv_width});
    fixture.add(prefix + "attn_gate.weight", {model.embedding_length, value_width});
    fixture.add(prefix + "ssm_conv1d.weight", {model.ssm_convolution_kernel, conv_width});
    fixture.add(prefix + "ssm_dt.bias", {model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_a", {model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_beta.weight", {model.embedding_length, model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_alpha.weight", {model.embedding_length, model.ssm_time_step_rank});
    fixture.add(prefix + "ssm_norm.weight", {model.ssm_state_size});
    fixture.add(prefix + "ssm_out.weight", {value_width, model.embedding_length});
}

void add_common(Fixture& fixture, const oracle::model::Qwen35Manifest& model, std::uint32_t block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    fixture.add(prefix + "attn_norm.weight", {model.embedding_length});
    fixture.add(prefix + "post_attention_norm.weight", {model.embedding_length});
    fixture.add(prefix + "ffn_gate.weight", {model.embedding_length, model.feed_forward_length});
    fixture.add(prefix + "ffn_down.weight", {model.feed_forward_length, model.embedding_length});
    fixture.add(prefix + "ffn_up.weight", {model.embedding_length, model.feed_forward_length});
}

Fixture make_base_fixture(const oracle::model::Qwen35Manifest& model, bool with_mtp) {
    Fixture fixture;
    fixture.add("token_embd.weight", {model.embedding_length, model.vocabulary_size});
    fixture.add("output_norm.weight", {model.embedding_length});
    for (std::uint32_t block = 0; block < model.backbone_block_count; ++block) {
        add_common(fixture, model, block);
        if (model.is_full_attention_block(block)) add_attention(fixture, model, block);
        else add_recurrent(fixture, model, block);
    }
    if (with_mtp) {
        const std::uint32_t mtp_block = model.backbone_block_count;
        add_common(fixture, model, mtp_block);
        add_attention(fixture, model, mtp_block);
        const std::string nextn = "blk." + std::to_string(mtp_block) + ".nextn.";
        fixture.add(nextn + "eh_proj.weight", {model.embedding_length * 2U, model.embedding_length});
        fixture.add(nextn + "enorm.weight", {model.embedding_length});
        fixture.add(nextn + "hnorm.weight", {model.embedding_length});
        fixture.add(nextn + "shared_head_norm.weight", {model.embedding_length});
    }
    return fixture;
}

// --- Fixture A: all-zero backbone + all-zero MTP. Every real forward
// argmaxes to token 0 regardless of input (embedding is zero too, so the
// residual stream is zero everywhere; RMSNorm of zero stays zero; the tied
// output head against a zero vector gives all-zero logits, and argmax's
// first-strictly-greater-wins tie-break picks index 0). MTP's own eh_proj is
// also zero, so it likewise always predicts 0. This gives a genuine,
// real-forward-verified FULL ACCEPT for any prompt/seed, at any depth. -----

oracle::model::Qwen35Weights make_fixture_zero(const oracle::model::Qwen35Manifest& model, Fixture& storage) {
    storage = make_base_fixture(model, true);
    return oracle::model::bind_qwen35_weights(storage.views, model);
}

// --- Fixture B: "identity" backbone + "identity" MTP. token_embd is
// one-hot per token (requires embedding_length == vocabulary_size);
// output_norm is uniform, so RMSNorm rescales without changing shape; every
// block's attn_output/ffn_down stay zero (pure residual passthrough), so
// decoding token T always argmaxes to T (self-referential). eh_proj is an
// identity matrix on the e_norm/candidate-token half of the concat (zero on
// the h_norm half) -- verified against reference_mapped_tensor_matvec's
// exact row-major convention (quantized_reference.cpp): flat index for
// (output_row=j, input_col=i) is j*(2n)+i, so eh_identity[j*2n+j]=1 for
// j in [0,n) gives eh_proj_output == e_norm exactly. This makes MTP predict
// its own candidate token, so a chained draft is [T,T,T,...] for any seed T
// -- matching the backbone's own self-referential continuation exactly, a
// genuine, real-forward-verified FULL ACCEPT for *any* token, used here to
// exercise real UTF-8/reasoning-boundary token IDs from the shared
// byte-fallback tokenizer. -------------------------------------------------

oracle::model::Qwen35Weights make_fixture_identity(const oracle::model::Qwen35Manifest& model,
                                                    Fixture& storage) {
    storage = make_base_fixture(model, true);
    const std::size_t n = model.embedding_length;
    const std::size_t v = model.vocabulary_size;
    require(n == v, "identity fixture requires embedding_length == vocabulary_size");

    std::vector<float> embed(n * v, 0.0F);
    for (std::size_t token = 0; token < v; ++token) embed[token * n + token] = 1.0F;
    storage.set_f32("token_embd.weight", embed);
    storage.set_f32("output_norm.weight", std::vector<float>(n, 1.0F));

    std::vector<float> eh(2 * n * n, 0.0F);
    for (std::size_t j = 0; j < n; ++j) eh[j * (2 * n) + j] = 1.0F;
    const std::uint32_t mtp_block = model.backbone_block_count;
    const std::string nextn = "blk." + std::to_string(mtp_block) + ".nextn.";
    storage.set_f32(nextn + "eh_proj.weight", eh);
    storage.set_f32(nextn + "enorm.weight", std::vector<float>(n, 1.0F));
    storage.set_f32(nextn + "hnorm.weight", std::vector<float>(n, 1.0F));
    storage.set_f32(nextn + "shared_head_norm.weight", std::vector<float>(n, 1.0F));

    return oracle::model::bind_qwen35_weights(storage.views, model);
}

// --- Fixture C: "textured" -- identity-shaped token_embd/output_norm (as
// fixture B) but MTP's eh_proj is a small fixed (non-random, deterministic)
// nonzero pattern instead of an exact identity, so its prediction is
// *usually* wrong (a real, verified ZERO ACCEPT for most seeds) but happens
// to agree with the self-referential backbone for specific seeds -- 5 and
// 28 in this construction, empirically discovered and reproducible, giving
// a real, verified PARTIAL ACCEPT (accepted=1, rejected=1) at depth 3. See
// docs/PHASE_2F.md ("Slice 6") for the exact discovery methodology (mirrors
// this project's established pattern of empirically searching a
// deterministic-but-nontrivial fixture rather than hand-deriving real-model
// behavior, e.g. tests/test_phase2e.cpp's discover_natural_tokens). -------

oracle::model::Qwen35Weights make_fixture_textured(const oracle::model::Qwen35Manifest& model,
                                                    Fixture& storage) {
    storage = make_base_fixture(model, true);
    const std::size_t n = model.embedding_length;
    const std::size_t v = model.vocabulary_size;
    require(n == v, "textured fixture requires embedding_length == vocabulary_size");

    std::vector<float> embed(n * v, 0.0F);
    for (std::size_t token = 0; token < v; ++token) embed[token * n + token] = 1.0F;
    storage.set_f32("token_embd.weight", embed);
    storage.set_f32("output_norm.weight", std::vector<float>(n, 1.0F));

    std::vector<float> eh(2 * n * n);
    for (std::size_t row = 0; row < 2 * n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            eh[row * n + col] = 0.1F * std::sin(static_cast<float>(row * n + col) * 0.37F + 1.0F);
        }
    }
    const std::uint32_t mtp_block = model.backbone_block_count;
    const std::string nextn = "blk." + std::to_string(mtp_block) + ".nextn.";
    storage.set_f32(nextn + "eh_proj.weight", eh);
    storage.set_f32(nextn + "enorm.weight", std::vector<float>(n, 1.0F));
    storage.set_f32(nextn + "hnorm.weight", std::vector<float>(n, 1.0F));
    storage.set_f32(nextn + "shared_head_norm.weight", std::vector<float>(n, 1.0F));

    return oracle::model::bind_qwen35_weights(storage.views, model);
}

// --- Non-MTP fixture (for the "MTP requested on a non-MTP model" test). --

oracle::model::Qwen35Weights make_fixture_no_mtp(const oracle::model::Qwen35Manifest& model,
                                                  Fixture& storage) {
    storage = make_base_fixture(model, false);
    return oracle::model::bind_qwen35_weights(storage.views, model);
}

// --- byte-fallback tokenizer fixture, adapted from
// tests/test_phase2e.cpp's write_tokenizer_fixture/make_text_tokenizer. ---

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
void write_key_string_array(std::ostream& output, std::string_view key,
                            const std::vector<std::string>& values) {
    write_string(output, key);
    write_u32(output, 9);
    write_u32(output, 8);
    write_u64(output, values.size());
    for (const std::string& value : values) write_string(output, value);
}
void write_key_i32_array(std::ostream& output, std::string_view key,
                         const std::vector<std::int32_t>& values) {
    write_string(output, key);
    write_u32(output, 9);
    write_u32(output, 5);
    write_u64(output, values.size());
    for (std::int32_t value : values) write_i32(output, value);
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
// token for byte value N (mirrors test_phase2e.cpp's proven fixture).
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
    oracle::tokenizer::TokenId merged_e_acute{0};  // one token spanning raw bytes 0xC3 0xA9 ('e' acute)

    TokenizerFixtureVocabulary() {
        tokens = byte_tokens();
        types.assign(tokens.size(), 1);  // TokenType::normal
        const auto add = [this](std::string token, std::int32_t type) {
            tokens.push_back(std::move(token));
            types.push_back(type);
            return static_cast<oracle::tokenizer::TokenId>(tokens.size() - 1);
        };
        merged_e_acute = add(tokens[195] + tokens[169], 1);
        // Pad out to a round vocabulary size so it can be shared with the
        // one-hot identity/textured model fixtures below.
        while (tokens.size() < 260) add("<|pad" + std::to_string(tokens.size()) + "|>", 5);
    }
};

[[nodiscard]] std::filesystem::path write_tokenizer_fixture() {
    const TokenizerFixtureVocabulary vocabulary;
    const auto path =
        std::filesystem::temp_directory_path() / "oracle-phase2f-slice6-tokenizer.gguf";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "failed to create Phase 2F Slice 6 tokenizer fixture");

    output.write("GGUF", 4);
    write_u32(output, 3);
    write_u64(output, 0);
    write_u64(output, 5);

    write_key_string(output, "tokenizer.ggml.model", "gpt2");
    write_key_string(output, "tokenizer.ggml.pre", "qwen35");
    write_key_string_array(output, "tokenizer.ggml.tokens", vocabulary.tokens);
    write_key_i32_array(output, "tokenizer.ggml.token_type", vocabulary.types);
    write_key_string_array(output, "tokenizer.ggml.merges", {});

    const std::uint64_t descriptor_end = static_cast<std::uint64_t>(output.tellp());
    const std::uint64_t data_offset = (descriptor_end + 31ULL) & ~31ULL;
    for (std::uint64_t offset = descriptor_end; offset < data_offset; ++offset) output.put('\0');

    output.close();
    require(static_cast<bool>(output), "failed while writing Phase 2F Slice 6 tokenizer fixture");
    return path;
}

[[nodiscard]] oracle::tokenizer::Qwen35Tokenizer make_text_tokenizer() {
    const std::filesystem::path path = write_tokenizer_fixture();
    const oracle::model::GgufFile file = oracle::model::GgufReader::read(path);
    std::filesystem::remove(path);
    return oracle::tokenizer::Qwen35Tokenizer(file);
}

const oracle::tokenizer::Qwen35Tokenizer& shared_text_tokenizer() {
    static const oracle::tokenizer::Qwen35Tokenizer tokenizer = make_text_tokenizer();
    return tokenizer;
}

// TokenizerFixtureVocabulary().merged_e_acute is deterministic (byte_tokens()
// always yields exactly 256 entries, and merged_e_acute is always the very
// next one appended), but computed via a real instance rather than a
// hand-counted literal to avoid exactly the kind of off-by-one this comment
// warns about.
const oracle::tokenizer::TokenId kMergedEAcute = TokenizerFixtureVocabulary().merged_e_acute;
constexpr oracle::tokenizer::TokenId kLoneLeadByte = 195;  // 0xC3 alone: invalid on repetition

}  // namespace

// ===========================================================================
// 1. MTP disabled is default; existing Phase 2E fixture unchanged.
// ===========================================================================

void test_mtp_disabled_by_default() {
    require(oracle::runtime::Qwen35GenerationRequest{}.mtp.mode ==
                oracle::runtime::Qwen35MtpMode::disabled,
            "MTP mode must default to disabled");
}

void test_mtp_off_matches_default_request_shape() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 5;
    request.sampling.temperature = 0.0F;
    // request.mtp left default (disabled) -- existing callers compile and
    // behave unchanged.
    const auto result = session.generate_fresh(request);
    require(!result.mtp_diagnostics.mtp_enabled, "diagnostics must report mtp disabled");
    require(result.mtp_diagnostics.draft_opportunities == 0, "no draft opportunities when disabled");
}

// ===========================================================================
// 3/4. MTP requested on a non-MTP model fails clearly; invalid draft depth.
// ===========================================================================

void test_mtp_on_non_mtp_model_fails_clearly() {
    const auto model = manifest(32, 64, 2, /*with_mtp=*/false);
    Fixture storage;
    const auto weights = make_fixture_no_mtp(model, storage);
    const auto& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.0F;
    request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    request.mtp.max_draft_depth = 1;
    require_throws([&] { static_cast<void>(session.generate_fresh(request)); }, "no MTP block");
}

void test_mtp_invalid_draft_depth_rejected() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);

    for (const std::uint32_t bad_depth : {0U, 4U, 10U}) {
        oracle::runtime::Qwen35GenerationRequest request;
        request.prompt_tokens = {0};
        request.max_generated_tokens = 3;
        request.sampling.temperature = 0.0F;
        request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
        request.mtp.max_draft_depth = bad_depth;
        require_throws([&] { static_cast<void>(session.generate_fresh(request)); },
                       "max_draft_depth must be 1, 2, or 3");
    }
}

// ===========================================================================
// 5. Greedy-only guard.
// ===========================================================================

void test_mtp_requires_greedy_sampling() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();
    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {0};
    request.max_generated_tokens = 3;
    request.sampling.temperature = 0.5F;
    request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    request.mtp.max_draft_depth = 2;
    require_throws([&] { static_cast<void>(session.generate_fresh(request)); }, "greedy sampling");
}

// ===========================================================================
// Lane A/B comparison helper: MTP OFF vs MTP ON, full canonical parity.
// ===========================================================================

struct LaneResult {
    oracle::runtime::Qwen35GenerationResult result;
    oracle::runtime::HybridCache state;
};

void compare_off_on(const oracle::model::Qwen35Manifest& model,
                    const oracle::model::Qwen35Weights& weights,
                    const oracle::tokenizer::Qwen35Tokenizer& tokenizer,
                    oracle::runtime::Qwen35GenerationRequest request_off,
                    oracle::runtime::Qwen35GenerationRequest request_on, std::size_t capacity,
                    std::string_view label) {
    request_off.mtp.mode = oracle::runtime::Qwen35MtpMode::disabled;

    std::vector<oracle::runtime::Qwen35GenerationEvent> events_off;
    std::vector<oracle::runtime::Qwen35GenerationEvent> events_on;

    oracle::runtime::Qwen35GenerationSession session_off(model, weights, tokenizer, capacity);
    const auto result_off = session_off.generate_fresh(
        request_off, [&](const oracle::runtime::Qwen35GenerationEvent& event) { events_off.push_back(event); });

    oracle::runtime::Qwen35GenerationSession session_on(model, weights, tokenizer, capacity);
    const auto result_on = session_on.generate_fresh(
        request_on, [&](const oracle::runtime::Qwen35GenerationEvent& event) { events_on.push_back(event); });

    const std::string tag = std::string(label) + ": ";
    require(result_off.finish_reason == result_on.finish_reason, tag + "finish_reason mismatch");
    require(result_off.generated_tokens.size() == result_on.generated_tokens.size(),
            tag + "generated token count mismatch");
    for (std::size_t i = 0; i < result_off.generated_tokens.size(); ++i) {
        require(result_off.generated_tokens[i].token_id == result_on.generated_tokens[i].token_id,
                tag + "generated token id mismatch at index " + std::to_string(i));
        require(result_off.generated_tokens[i].position == result_on.generated_tokens[i].position,
                tag + "generated token position mismatch at index " + std::to_string(i));
    }
    require(result_off.generated_text == result_on.generated_text, tag + "generated_text mismatch");
    require(result_off.final_sequence_length == result_on.final_sequence_length,
            tag + "final_sequence_length mismatch");
    require(events_off.size() == events_on.size(), tag + "callback event count mismatch");
    for (std::size_t i = 0; i < events_off.size(); ++i) {
        require(events_off[i].token_id == events_on[i].token_id,
                tag + "event token_id mismatch at index " + std::to_string(i));
        require(events_off[i].generated_index == events_on[i].generated_index,
                tag + "event generated_index mismatch at index " + std::to_string(i));
        require(events_off[i].text_fragment == events_on[i].text_fragment,
                tag + "event text_fragment mismatch at index " + std::to_string(i));
        require(events_off[i].token_bytes == events_on[i].token_bytes,
                tag + "event token_bytes mismatch at index " + std::to_string(i));
        require(events_off[i].eos == events_on[i].eos, tag + "event eos flag mismatch at index " +
                                                            std::to_string(i));
    }

    const auto fp_off = oracle::runtime::fingerprint_qwen35_state(session_off.state());
    const auto fp_on = oracle::runtime::fingerprint_qwen35_state(session_on.state());
    require(fp_off == fp_on, tag + "final HybridCache fingerprint mismatch");
}

// ===========================================================================
// 6. full-accept decision batch (+ 24. deterministic repeat, 25. HybridCache
//    equality, 22. callback ordering, 9. exactly-once commit).
// ===========================================================================

void test_full_accept_decision_batch() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {1};
    base.max_generated_tokens = 6;
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "full-accept");

    // Diagnostics + exactly-once commit: with 6 tokens generated at depth 3,
    // there are exactly 2 draft opportunities, both fully accepted.
    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    std::size_t event_count = 0;
    const auto result = session.generate_fresh(
        on, [&](const oracle::runtime::Qwen35GenerationEvent&) { ++event_count; });
    require(event_count == result.generated_tokens.size(),
            "every committed token must produce exactly one ordinary event");
    require(result.mtp_diagnostics.mtp_enabled, "diagnostics must report mtp enabled");
    require(result.mtp_diagnostics.draft_opportunities == 2, "expected 2 draft opportunities");
    require(result.mtp_diagnostics.drafts_proposed == 6, "expected 6 total drafts proposed");
    require(result.mtp_diagnostics.drafts_accepted == 6, "expected all 6 drafts accepted");
    require(result.mtp_diagnostics.drafts_rejected == 0, "expected 0 rejected");
    require(result.mtp_diagnostics.unused_drafts == 0, "expected 0 unused");
    require(result.mtp_diagnostics.full_accept_chains == 2, "expected 2 full-accept chains");
    require(result.mtp_diagnostics.partial_accept_chains == 0, "expected 0 partial-accept chains");
    require(result.mtp_diagnostics.zero_accept_chains == 0, "expected 0 zero-accept chains");
}

void test_deterministic_repeat() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {1};
    request.max_generated_tokens = 7;
    request.sampling.temperature = 0.0F;
    request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    request.mtp.max_draft_depth = 3;

    oracle::runtime::Qwen35GenerationSession session_a(model, weights, tokenizer, 32);
    const auto result_a = session_a.generate_fresh(request);
    oracle::runtime::Qwen35GenerationSession session_b(model, weights, tokenizer, 32);
    const auto result_b = session_b.generate_fresh(request);

    require(result_a.generated_tokens.size() == result_b.generated_tokens.size(),
            "repeated runs must produce the same token count");
    for (std::size_t i = 0; i < result_a.generated_tokens.size(); ++i) {
        require(result_a.generated_tokens[i].token_id == result_b.generated_tokens[i].token_id,
                "repeated runs must produce identical token ids");
    }
    const auto fp_a = oracle::runtime::fingerprint_qwen35_state(session_a.state());
    const auto fp_b = oracle::runtime::fingerprint_qwen35_state(session_b.state());
    require(fp_a == fp_b, "repeated runs must produce identical final state");
}

// ===========================================================================
// 8/10/11. zero-accept decision batch; rejected/unused drafts emit no
//          ordinary token events.
// ===========================================================================

void test_zero_accept_decision_batch() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_textured(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {0};
    base.max_generated_tokens = 3;
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "zero-accept");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    const auto result = session.generate_fresh(
        on, [&](const oracle::runtime::Qwen35GenerationEvent& event) { events.push_back(event); });
    require(events.size() == result.generated_tokens.size(),
            "rejected/unused drafts must never produce an ordinary event beyond the committed ledger");
    require(result.mtp_diagnostics.zero_accept_chains >= 1, "expected at least one zero-accept chain");
    // Draft token 8 (from the depth-3 chain at seed 0) is never accepted and
    // must never appear as a committed event.
    for (const auto& event : events) {
        require(event.token_id == 0, "seed=0's zero-accept chain must only ever commit token 0");
    }
}

// ===========================================================================
// 7. partial-accept decision batch.
// ===========================================================================

void test_partial_accept_decision_batch() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_textured(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {5};
    // >= 3 so the first draft opportunity's effective depth is not clipped
    // below the requested 3 (see docs/PHASE_2F.md "Slice 6" depth bounding):
    // the empirically-discovered seed=5 pattern (accepted=1, rejected=1,
    // unused=1) only appears at depth 3.
    base.max_generated_tokens = 3;
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "partial-accept");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(result.mtp_diagnostics.partial_accept_chains >= 1,
            "expected at least one partial-accept chain for seed=5 at depth 3");
    require(result.mtp_diagnostics.drafts_rejected >= 1, "expected at least one rejected draft");
    require(result.mtp_diagnostics.unused_drafts >= 1,
            "expected at least one unused (discarded) draft for seed=5 at depth 3");
}

// ===========================================================================
// 12. EOS inside a verified MTP decision batch.
// ===========================================================================

void test_eos_inside_verified_prefix() {
    auto model = manifest(32, 32);
    model.eos_token_id = 5;  // matches seed=5's partial-accept fixture below
    Fixture storage;
    const auto weights = make_fixture_textured(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {5};
    base.max_generated_tokens = 10;  // generous budget; EOS must still stop early
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "eos-in-prefix");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::eos,
            "EOS must still terminate generation under MTP");
    require(result.generated_tokens.size() == 1,
            "EOS committed at the first opportunity must leave exactly one generated token");
    require(result.generated_tokens.back().token_id == 5, "the committed token must be EOS (5)");
}

// ===========================================================================
// 13. token stop inside a verified MTP decision batch.
// ===========================================================================

void test_token_stop_inside_verified_prefix() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {1};
    base.max_generated_tokens = 6;
    base.sampling.temperature = 0.0F;
    base.token_stop_sequences = {{0, 0}};  // matches after exactly 2 committed zeros

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;  // proposes 3, only 2 should ever commit

    compare_off_on(model, weights, tokenizer, base, on, 32, "token-stop-in-prefix");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "token stop must still terminate generation under MTP");
    require(result.generated_tokens.size() == 2,
            "the verified 3rd draft must be discarded once the stop matches at 2");
    require(result.stop_match.has_value(), "stop_match must be populated");
    require(result.stop_match->kind == oracle::runtime::Qwen35StopKind::token_sequence,
            "stop kind must be token_sequence");
}

// ===========================================================================
// 14. text stop inside a verified MTP decision batch.
// ===========================================================================

void test_text_stop_inside_verified_prefix() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {1};
    base.max_generated_tokens = 6;
    base.sampling.temperature = 0.0F;
    // Token 0's own decoded text is whatever byte_tokens()[0] maps to; find
    // it once via the tokenizer and build a 2-repeat text stop from it so
    // the test is self-consistent regardless of the exact byte mapping.
    const std::array<oracle::tokenizer::TokenId, 1> single{0};
    const std::string one_token_text = tokenizer.decode(single, oracle::tokenizer::DecodeOptions{});
    base.text_stop_sequences = {one_token_text + one_token_text};

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "text-stop-in-prefix");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::stop_sequence,
            "text stop must still terminate generation under MTP");
    require(result.stop_match.has_value() &&
                result.stop_match->kind == oracle::runtime::Qwen35StopKind::text_sequence,
            "stop kind must be text_sequence");
    require(result.generated_tokens.size() == 2,
            "the verified 3rd draft must be discarded once the text stop matches at 2 tokens");
}

// ===========================================================================
// 15/16. UTF-8 split across verified MTP tokens; malformed UTF-8 parity.
// ===========================================================================

void test_utf8_valid_multibyte_under_mtp() {
    auto model = manifest(260, 260);
    Fixture storage;
    const auto weights = make_fixture_identity(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {kMergedEAcute};  // one token whose own bytes are the
                                           // complete 2-byte UTF-8 'e' acute
    base.max_generated_tokens = 3;
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 300, "utf8-valid");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 300);
    const auto result = session.generate_fresh(on);
    require(result.generated_text == "\xC3\xA9\xC3\xA9\xC3\xA9",
            "three repeated merged e-acute tokens must decode to three valid 'e' acutes, "
            "fed through the assembler one committed token at a time under MTP");
    require(result.mtp_diagnostics.full_accept_chains >= 1, "expected a full-accept chain");
}

void test_utf8_malformed_parity_under_mtp() {
    auto model = manifest(260, 260);
    Fixture storage;
    const auto weights = make_fixture_identity(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {kLoneLeadByte};  // 0xC3 repeated: never a valid sequence
    base.max_generated_tokens = 3;
    base.sampling.temperature = 0.0F;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 300, "utf8-malformed");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 300);
    const auto result = session.generate_fresh(on);
    // Every 0xC3 after the first looks like a plausible 2-byte lead but is
    // immediately followed by another 0xC3 (not a valid continuation byte),
    // so each is individually replaced.
    require(result.generated_text.find("\xEF\xBF\xBD") != std::string::npos,
            "malformed UTF-8 must still be replaced with U+FFFD under MTP");
}

// ===========================================================================
// 17. reasoning boundary parity.
// ===========================================================================

void test_reasoning_boundary_parity_under_mtp() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_identity(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {3};
    boundary.end_tokens = {7};
    boundary.force_close_supported = false;

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {3};  // self-referential: generates 3,3,3,... forever
    base.max_generated_tokens = 4;
    base.sampling.temperature = 0.0F;
    base.reasoning_boundary = boundary;
    base.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::off;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "reasoning-boundary");

    oracle::runtime::Qwen35GenerationSession session_off(model, weights, tokenizer, 32);
    const auto result_off = session_off.generate_fresh(base);
    oracle::runtime::Qwen35GenerationSession session_on(model, weights, tokenizer, 32);
    const auto result_on = session_on.generate_fresh(on);
    require(result_off.reasoning_active_at_finish == result_on.reasoning_active_at_finish,
            "reasoning_active_at_finish must match between MTP OFF/ON");
}

// ===========================================================================
// 18. reasoning-loop stop parity.
// ===========================================================================

void test_reasoning_loop_stop_parity_under_mtp() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_identity(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {3};
    boundary.end_tokens = {9};  // never naturally generated by this self-referential fixture
    boundary.force_close_supported = false;

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {3};
    base.max_generated_tokens = 10;
    base.sampling.temperature = 0.0F;
    base.reasoning_boundary = boundary;
    base.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
    base.reasoning_loop.minimum_reasoning_tokens = 3;
    base.reasoning_loop.inspection_window = 8;
    base.reasoning_loop.maximum_period = 2;
    base.reasoning_loop.minimum_repeated_coverage = 3;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "reasoning-loop-stop");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::reasoning_loop,
            "reasoning-loop stop must still terminate generation under MTP");
    require(result.generated_tokens.size() == 3,
            "must stop exactly when the loop is confirmed, discarding any verified suffix");
}

// ===========================================================================
// 19. reasoning-loop force-close parity.
// ===========================================================================

void test_reasoning_loop_force_close_parity_under_mtp() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_identity(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35ReasoningBoundary boundary;
    boundary.start_tokens = {3};
    boundary.end_tokens = {9};
    boundary.force_close_supported = true;

    oracle::runtime::Qwen35GenerationRequest base;
    base.prompt_tokens = {3};
    base.max_generated_tokens = 10;
    base.sampling.temperature = 0.0F;
    base.reasoning_boundary = boundary;
    base.reasoning_loop.policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
    base.reasoning_loop.minimum_reasoning_tokens = 3;
    base.reasoning_loop.inspection_window = 8;
    base.reasoning_loop.maximum_period = 2;
    base.reasoning_loop.minimum_repeated_coverage = 3;

    auto on = base;
    on.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    on.mtp.max_draft_depth = 3;

    compare_off_on(model, weights, tokenizer, base, on, 32, "reasoning-loop-force-close");

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(on);
    require(!result.reasoning_interventions.empty(), "at least one intervention must be recorded");
    require(result.reasoning_interventions.front().closure_succeeded,
            "force-close must genuinely succeed via the real target forward, unaffected by MTP");
    // The forced end token (9) must appear in the canonical ledger.
    bool saw_forced = false;
    for (const auto& token : result.generated_tokens) {
        if (token.token_id == 9) saw_forced = true;
    }
    require(saw_forced, "the forced reasoning-end token must be genuinely committed");
}

// ===========================================================================
// 20. max_tokens depth clipping.
// ===========================================================================

void test_max_tokens_depth_clipping() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    for (const std::size_t remaining : {std::size_t{1}, std::size_t{2}}) {
        oracle::runtime::Qwen35GenerationRequest request;
        request.prompt_tokens = {1};
        request.max_generated_tokens = remaining;
        request.sampling.temperature = 0.0F;
        request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
        request.mtp.max_draft_depth = 3;  // requested depth exceeds remaining budget

        oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
        const auto result = session.generate_fresh(request);
        require(result.generated_tokens.size() == remaining,
                "generated token count must never exceed max_generated_tokens=" +
                    std::to_string(remaining));
        require(result.finish_reason == oracle::runtime::Qwen35FinishReason::max_tokens,
                "must terminate via max_tokens, not overshoot via MTP");
    }
}

// ===========================================================================
// 21. context-capacity depth clipping.
// ===========================================================================

void test_context_capacity_depth_clipping() {
    const auto model = manifest(32, 64);
    Fixture storage;
    const auto weights = make_fixture_zero(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    // Capacity 4: prompt occupies 1, leaving room for only 3 more positions.
    const std::size_t capacity = 4;
    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {1};
    request.max_generated_tokens = 100;  // budget alone would not stop early
    request.sampling.temperature = 0.0F;
    request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    request.mtp.max_draft_depth = 3;

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, capacity);
    const auto result = session.generate_fresh(request);
    require(result.final_sequence_length <= capacity,
            "final sequence length must never exceed context capacity, even via MTP");
    require(result.finish_reason == oracle::runtime::Qwen35FinishReason::context_exhausted,
            "must terminate via context_exhausted, not overrun via MTP");

    // Lane B comparison against MTP OFF at the same capacity.
    auto off_request = request;
    off_request.mtp.mode = oracle::runtime::Qwen35MtpMode::disabled;
    compare_off_on(model, weights, tokenizer, off_request, request, capacity, "context-capacity");
}

// ===========================================================================
// 23. MTP diagnostic accounting invariants.
// ===========================================================================

void test_mtp_diagnostics_accounting_invariants() {
    const auto model = manifest(32, 32);
    Fixture storage;
    const auto weights = make_fixture_textured(model, storage);
    const auto& tokenizer = shared_text_tokenizer();

    oracle::runtime::Qwen35GenerationRequest request;
    request.prompt_tokens = {5};
    request.max_generated_tokens = 4;
    request.sampling.temperature = 0.0F;
    request.mtp.mode = oracle::runtime::Qwen35MtpMode::reference;
    request.mtp.max_draft_depth = 3;

    oracle::runtime::Qwen35GenerationSession session(model, weights, tokenizer, 32);
    const auto result = session.generate_fresh(request);
    const auto& diag = result.mtp_diagnostics;

    require(diag.mtp_enabled, "diagnostics must report mtp enabled");
    require(diag.requested_draft_depth == 3, "diagnostics must record the requested depth");
    require(diag.full_accept_chains + diag.partial_accept_chains + diag.zero_accept_chains ==
                diag.draft_opportunities,
            "chain classification counts must sum to draft_opportunities");
    require(diag.drafts_accepted + diag.drafts_rejected <= diag.drafts_proposed,
            "accepted+rejected must never exceed proposed, summed across the whole run");
}

int main() {
    try {
        test_mtp_disabled_by_default();
        test_mtp_off_matches_default_request_shape();
        test_mtp_on_non_mtp_model_fails_clearly();
        test_mtp_invalid_draft_depth_rejected();
        test_mtp_requires_greedy_sampling();
        test_full_accept_decision_batch();
        test_deterministic_repeat();
        test_zero_accept_decision_batch();
        test_partial_accept_decision_batch();
        test_eos_inside_verified_prefix();
        test_token_stop_inside_verified_prefix();
        test_text_stop_inside_verified_prefix();
        test_utf8_valid_multibyte_under_mtp();
        test_utf8_malformed_parity_under_mtp();
        test_reasoning_boundary_parity_under_mtp();
        test_reasoning_loop_stop_parity_under_mtp();
        test_reasoning_loop_force_close_parity_under_mtp();
        test_max_tokens_depth_clipping();
        test_context_capacity_depth_clipping();
        test_mtp_diagnostics_accounting_invariants();
        std::cout << "Phase 2F Slice 6 guarded generation-session integration tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2F Slice 6 test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
