#include "oracle/model/gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/runtime/qwen35_chat.hpp"
#include "oracle/tokenizer/qwen35_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

class TestRunner {
public:
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAIL: " << message << '\n';
        }
    }

    template <typename Function>
    void expect_throws(Function&& function, std::string_view message) {
        try {
            std::invoke(std::forward<Function>(function));
            ++failures_;
            std::cerr << "FAIL: " << message << " (no exception)\n";
        } catch (const std::exception&) {
        }
    }

    [[nodiscard]] int result() const noexcept { return failures_ == 0 ? 0 : 1; }
    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_{0};
};

template <typename Unsigned>
void write_unsigned_le(std::ostream& output, Unsigned value) {
    static_assert(std::is_unsigned_v<Unsigned>);
    for (std::size_t index = 0; index < sizeof(Unsigned); ++index) {
        output.put(static_cast<char>((value >> (index * 8U)) & 0xffU));
    }
}

void write_u32(std::ostream& output, std::uint32_t value) {
    write_unsigned_le(output, value);
}
void write_u64(std::ostream& output, std::uint64_t value) {
    write_unsigned_le(output, value);
}
void write_i32(std::ostream& output, std::int32_t value) {
    write_u32(output, std::bit_cast<std::uint32_t>(value));
}
void write_f32(std::ostream& output, float value) {
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
void write_key_f32(std::ostream& output, std::string_view key, float value) {
    write_string(output, key);
    write_u32(output, 6);
    write_f32(output, value);
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

[[nodiscard]] std::vector<std::string> byte_tokens() {
    std::vector<std::string> output(256);
    std::array<bool, 256> direct{};
    for (std::uint32_t byte = 33; byte <= 126; ++byte) {
        direct[byte] = true;
    }
    for (std::uint32_t byte = 161; byte <= 172; ++byte) {
        direct[byte] = true;
    }
    for (std::uint32_t byte = 174; byte <= 255; ++byte) {
        direct[byte] = true;
    }
    std::uint32_t extension = 0;
    for (std::uint32_t byte = 0; byte <= 255; ++byte) {
        output[byte] = utf8_from_codepoint(direct[byte] ? byte : 256U + extension++);
    }
    return output;
}

struct FixtureVocabulary {
    std::vector<std::string> tokens;
    std::vector<std::int32_t> types;
    std::vector<std::string> merges;
    std::uint32_t im_start{0};
    std::uint32_t im_end{0};
    std::uint32_t vision_pad{0};
};

[[nodiscard]] FixtureVocabulary make_vocabulary() {
    FixtureVocabulary vocabulary;
    vocabulary.tokens = byte_tokens();
    vocabulary.types.assign(vocabulary.tokens.size(), 1);

    const auto add = [&vocabulary](std::string token, std::int32_t type = 1) {
        if (std::ranges::find(vocabulary.tokens, token) != vocabulary.tokens.end()) {
            throw std::runtime_error("duplicate fixture token");
        }
        vocabulary.tokens.push_back(std::move(token));
        vocabulary.types.push_back(type);
        return static_cast<std::uint32_t>(vocabulary.tokens.size() - 1);
    };

    for (std::string token : {"he", "hel", "hell", "hello", "Ġw", "Ġwo",
                              "Ġwor", "Ġworl", "Ġworld", "oĠ", "12"}) {
        static_cast<void>(add(std::move(token)));
    }
    vocabulary.merges = {"h e",      "he l",   "hel l", "hell o",
                         "Ġ w",      "Ġw o",   "Ġwo r", "Ġwor l",
                         "Ġworl d",  "o Ġ",    "1 2"};

    static_cast<void>(add("<|endoftext|>", 3));
    vocabulary.im_start = add("<|im_start|>", 3);
    vocabulary.im_end = add("<|im_end|>", 3);
    vocabulary.vision_pad = add("<|vision_pad|>", 3);
    static_cast<void>(add("<tool_call>", 4));
    static_cast<void>(add("</tool_call>", 4));
    static_cast<void>(add("<tool_response>", 4));
    static_cast<void>(add("</tool_response>", 4));
    static_cast<void>(add("<think>", 4));
    static_cast<void>(add("</think>", 4));
    return vocabulary;
}

struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint64_t offset{0};
};

[[nodiscard]] std::filesystem::path write_qwen35_fixture(bool mtp) {
    const FixtureVocabulary vocabulary = make_vocabulary();
    const auto path = std::filesystem::temp_directory_path() /
                      (mtp ? "oracle-phase1d-mtp.gguf" : "oracle-phase1d-base.gguf");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create Phase 1D fixture");
    }

    const std::uint64_t tensor_count = mtp ? 2 : 1;
    const std::uint64_t metadata_count = mtp ? 31 : 30;
    output.write("GGUF", 4);
    write_u32(output, 3);
    write_u64(output, tensor_count);
    write_u64(output, metadata_count);

    write_key_string(output, "general.architecture", "qwen35");
    write_key_string(output, "general.name", mtp ? "fixture-mtp" : "fixture-base");
    write_key_u32(output, "general.alignment", 32);
    write_key_u32(output, "qwen35.block_count", mtp ? 5 : 4);
    write_key_u32(output, "qwen35.context_length", 1024);
    write_key_u32(output, "qwen35.embedding_length", 8);
    write_key_u32(output, "qwen35.feed_forward_length", 16);
    write_key_u32(output, "qwen35.attention.head_count", 4);
    write_key_u32(output, "qwen35.attention.head_count_kv", 2);
    write_key_i32_array(output, "qwen35.rope.dimension_sections", {1, 1, 0, 0});
    write_key_f32(output, "qwen35.rope.freq_base", 10000000.0F);
    write_key_f32(output, "qwen35.attention.layer_norm_rms_epsilon", 1.0e-6F);
    write_key_u32(output, "qwen35.attention.key_length", 4);
    write_key_u32(output, "qwen35.attention.value_length", 4);
    write_key_u32(output, "qwen35.ssm.conv_kernel", 4);
    write_key_u32(output, "qwen35.ssm.state_size", 8);
    write_key_u32(output, "qwen35.ssm.group_count", 2);
    write_key_u32(output, "qwen35.ssm.time_step_rank", 2);
    write_key_u32(output, "qwen35.ssm.inner_size", 16);
    write_key_u32(output, "qwen35.full_attention_interval", 4);
    write_key_u32(output, "qwen35.rope.dimension_count", 4);
    if (mtp) {
        write_key_u32(output, "qwen35.nextn_predict_layers", 1);
    }
    write_key_string(output, "tokenizer.ggml.model", "gpt2");
    write_key_string(output, "tokenizer.ggml.pre", "qwen35");
    write_key_string_array(output, "tokenizer.ggml.tokens", vocabulary.tokens);
    write_key_i32_array(output, "tokenizer.ggml.token_type", vocabulary.types);
    write_key_string_array(output, "tokenizer.ggml.merges", vocabulary.merges);
    write_key_u32(output, "tokenizer.ggml.eos_token_id", vocabulary.im_end);
    write_key_u32(output, "tokenizer.ggml.padding_token_id", vocabulary.vision_pad);
    write_key_string(output, "tokenizer.chat_template", "qwen35-fixture-template");
    write_key_u32(output, "general.quantization_version", 2);

    const std::uint64_t embedding_bytes =
        8ULL * static_cast<std::uint64_t>(vocabulary.tokens.size()) * 4ULL;
    std::vector<TensorDescriptor> tensors;
    tensors.push_back({"token_embd.weight",
                       {8, static_cast<std::uint64_t>(vocabulary.tokens.size())},
                       0});
    if (mtp) {
        const std::uint64_t next_offset = (embedding_bytes + 31ULL) & ~31ULL;
        tensors.push_back({"blk.4.nextn.eh_proj.weight", {8, 8}, next_offset});
    }

    for (const TensorDescriptor& tensor : tensors) {
        write_string(output, tensor.name);
        write_u32(output, static_cast<std::uint32_t>(tensor.dimensions.size()));
        for (std::uint64_t dimension : tensor.dimensions) {
            write_u64(output, dimension);
        }
        write_u32(output, 0);
        write_u64(output, tensor.offset);
    }

    const std::uint64_t descriptor_end = static_cast<std::uint64_t>(output.tellp());
    const std::uint64_t data_offset = (descriptor_end + 31ULL) & ~31ULL;
    for (std::uint64_t offset = descriptor_end; offset < data_offset; ++offset) {
        output.put('\0');
    }
    const std::uint64_t total_payload =
        mtp ? tensors.back().offset + 8ULL * 8ULL * 4ULL : embedding_bytes;
    for (std::uint64_t offset = 0; offset < total_payload; ++offset) {
        output.put('\0');
    }
    output.close();
    return path;
}

void test_manifests(TestRunner& runner,
                    const oracle::model::GgufFile& base,
                    const oracle::model::GgufFile& mtp) {
    const auto base_manifest = oracle::model::load_qwen35_manifest(base);
    runner.expect(base_manifest.total_block_count == 4, "base total block count");
    runner.expect(base_manifest.backbone_block_count == 4, "base backbone block count");
    runner.expect(!base_manifest.has_mtp(), "base has no MTP head");
    runner.expect(base_manifest.is_full_attention_block(3),
                  "fourth base block is full attention");
    runner.expect(!base_manifest.is_full_attention_block(2),
                  "third base block is SSM");
    runner.expect(base_manifest.vocabulary_size == make_vocabulary().tokens.size(),
                  "manifest vocabulary size");
    runner.expect(!base_manifest.bos_token_id.has_value(), "manifest does not invent BOS");

    const auto mtp_manifest = oracle::model::load_qwen35_manifest(mtp);
    runner.expect(mtp_manifest.total_block_count == 5, "MTP total block count");
    runner.expect(mtp_manifest.backbone_block_count == 4,
                  "MTP appended block excluded from backbone");
    runner.expect(mtp_manifest.nextn_predict_layers == 1, "MTP layer count");
    runner.expect(mtp_manifest.has_mtp(), "MTP manifest detected");
    runner.expect(oracle::model::qwen35_manifest_json(mtp_manifest).find(
                      "\"backbone_block_count\":4") != std::string::npos,
                  "manifest JSON contains resolved backbone count");

    oracle::model::GgufFile bad_architecture = base;
    for (auto& entry : bad_architecture.metadata) {
        if (entry.key == "general.architecture") {
            const auto* value = entry.value.get_if<std::string>();
            runner.expect(value != nullptr, "architecture fixture type");
            entry = {"general.architecture",
                     oracle::model::GgufValue(
                         oracle::model::GgufMetadataType::string,
                         std::string("not-qwen35"))};
            break;
        }
    }
    runner.expect_throws(
        [&bad_architecture] {
            static_cast<void>(oracle::model::load_qwen35_manifest(bad_architecture));
        },
        "manifest rejects wrong architecture");

    oracle::model::GgufFile missing_mtp_tensor = mtp;
    missing_mtp_tensor.tensors.erase(
        std::remove_if(missing_mtp_tensor.tensors.begin(),
                       missing_mtp_tensor.tensors.end(),
                       [](const auto& tensor) { return tensor.name.find(".nextn.") != std::string::npos; }),
        missing_mtp_tensor.tensors.end());
    runner.expect_throws(
        [&missing_mtp_tensor] {
            static_cast<void>(oracle::model::load_qwen35_manifest(missing_mtp_tensor));
        },
        "manifest rejects MTP metadata without appended nextn tensors");
}

void test_tokenizer(TestRunner& runner, const oracle::model::GgufFile& file) {
    const FixtureVocabulary fixture = make_vocabulary();
    const oracle::tokenizer::Qwen35Tokenizer tokenizer(file);
    runner.expect(tokenizer.vocabulary_size() == fixture.tokens.size(),
                  "tokenizer vocabulary loaded");
    runner.expect(tokenizer.merge_count() == fixture.merges.size(),
                  "tokenizer merge ranks loaded");
    runner.expect(tokenizer.eos_token_id() == fixture.im_end, "tokenizer EOS loaded");
    runner.expect(!tokenizer.bos_token_id().has_value(), "tokenizer does not invent BOS");
    runner.expect(tokenizer.chat_template() == "qwen35-fixture-template",
                  "chat template metadata loaded");

    const auto hello = tokenizer.find_token("hello");
    const auto world = tokenizer.find_token("Ġworld");
    runner.expect(hello.has_value() && world.has_value(), "merged fixture tokens exist");
    const std::vector<oracle::tokenizer::TokenId> ids = tokenizer.encode("hello world");
    runner.expect(ids.size() == 2 && hello && world && ids[0] == *hello && ids[1] == *world,
                  "Qwen35 byte BPE produces expected merged tokens");
    runner.expect(tokenizer.decode(ids) == "hello world", "ASCII tokenizer round trip");

    const std::string multilingual = "Café 中文 한글 🙂 e\xCC\x81";
    const auto multilingual_ids = tokenizer.encode(multilingual);
    runner.expect(tokenizer.decode(multilingual_ids) == multilingual,
                  "multilingual and combining-mark round trip");

    const auto number_ids = tokenizer.encode("12");
    const auto twelve = tokenizer.find_token("12");
    runner.expect(twelve.has_value(), "cross-boundary numeric merge fixture exists");
    runner.expect(number_ids.size() == 2 && twelve &&
                      std::ranges::find(number_ids, *twelve) == number_ids.end(),
                  "Qwen35 pre-tokenizer keeps digits as individual pieces");

    const std::string special_text = "hello<|im_start|>world";
    const auto ordinary = tokenizer.encode(special_text);
    const auto special = tokenizer.encode(
        special_text,
        oracle::tokenizer::EncodeOptions{.parse_special_tokens = true});
    runner.expect(std::ranges::find(ordinary, fixture.im_start) == ordinary.end(),
                  "ordinary encoding does not parse control tokens");
    runner.expect(std::ranges::find(special, fixture.im_start) != special.end(),
                  "special-token mode recognizes control tokens");
    runner.expect(tokenizer.decode(special) == special_text,
                  "special-token encoding round trip");
    runner.expect(tokenizer.decode(
                      special,
                      oracle::tokenizer::DecodeOptions{.skip_special_tokens = true}) ==
                      "helloworld",
                  "special-token decode can omit controls");

    oracle::model::GgufFile mismatched = file;
    for (auto& entry : mismatched.metadata) {
        if (entry.key == "tokenizer.ggml.token_type") {
            const auto* pointer =
                entry.value.get_if<std::shared_ptr<oracle::model::GgufArray>>();
            if (pointer != nullptr && *pointer != nullptr) {
                auto copy = std::make_shared<oracle::model::GgufArray>(**pointer);
                copy->values.pop_back();
                entry = {"tokenizer.ggml.token_type",
                         oracle::model::GgufValue(
                             oracle::model::GgufMetadataType::array,
                             std::move(copy))};
            }
            break;
        }
    }
    runner.expect_throws(
        [&mismatched] {
            static_cast<void>(oracle::tokenizer::Qwen35Tokenizer(mismatched));
        },
        "tokenizer rejects token/type length mismatch");
}

void test_chat(TestRunner& runner, const oracle::model::GgufFile& file) {
    oracle::runtime::Qwen35ChatRequest request;
    request.messages = {
        {oracle::runtime::ChatRole::system, "You are Lumina.", {}, {}},
        {oracle::runtime::ChatRole::user, "Check the engine.", {}, {}},
        {oracle::runtime::ChatRole::assistant,
         "",
         "I should inspect status.",
         {{"oracle_status", "{\"detail\": true}"}}},
        {oracle::runtime::ChatRole::tool, "{\"healthy\":true}", {}, {}},
        {oracle::runtime::ChatRole::tool, "{\"port\":5150}", {}, {}},
    };
    request.tools_json = {
        "{\"type\":\"function\",\"function\":{\"name\":\"oracle_status\"}}"};
    request.add_generation_prompt = true;

    const std::string prompt = oracle::runtime::format_qwen35_chat(request);
    runner.expect(prompt.starts_with("<|im_start|>system\nYou are Lumina.\n\n# Tools"),
                  "chat formatter emits tool-aware system message");
    runner.expect(prompt.find("<think>\nI should inspect status.\n</think>") !=
                      std::string::npos,
                  "chat formatter preserves assistant reasoning");
    runner.expect(prompt.find(
                      "<tool_call>\n{\"name\": \"oracle_status\", \"arguments\": "
                      "{\"detail\": true}}\n</tool_call>") != std::string::npos,
                  "chat formatter emits OpenAI-style tool call");
    runner.expect(prompt.find(
                      "<tool_response>\n{\"healthy\":true}\n</tool_response>\n"
                      "<tool_response>\n{\"port\":5150}\n</tool_response>") !=
                      std::string::npos,
                  "chat formatter groups adjacent tool responses");
    runner.expect(prompt.ends_with("<|im_start|>assistant\n<think>\n"),
                  "chat formatter adds Qwen35 generation prompt");

    const oracle::tokenizer::Qwen35Tokenizer tokenizer(file);
    const auto ids = tokenizer.encode(
        prompt,
        oracle::tokenizer::EncodeOptions{.parse_special_tokens = true});
    runner.expect(tokenizer.decode(ids) == prompt,
                  "formatted chat survives tokenizer round trip");

    runner.expect_throws(
        [] {
            static_cast<void>(oracle::runtime::format_qwen35_chat({}));
        },
        "chat formatter rejects empty conversations");
}

}  // namespace

int main() {
    TestRunner runner;
    const std::filesystem::path base_path = write_qwen35_fixture(false);
    const std::filesystem::path mtp_path = write_qwen35_fixture(true);
    try {
        const oracle::model::GgufFile base = oracle::model::GgufReader::read(base_path);
        const oracle::model::GgufFile mtp = oracle::model::GgufReader::read(mtp_path);
        test_manifests(runner, base, mtp);
        test_tokenizer(runner, base);
        test_chat(runner, base);
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT TEST ERROR: " << error.what() << '\n';
        std::filesystem::remove(base_path);
        std::filesystem::remove(mtp_path);
        return 1;
    }
    std::filesystem::remove(base_path);
    std::filesystem::remove(mtp_path);

    if (runner.failures() == 0) {
        std::cout << "all Phase 1D tests passed\n";
    }
    return runner.result();
}
