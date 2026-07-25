#include "oracle/model/gguf.hpp"
#include "oracle/model/mapped_gguf.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

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

void write_u8(std::ostream& output, std::uint8_t value) {
    write_unsigned_le(output, value);
}

void write_u32(std::ostream& output, std::uint32_t value) {
    write_unsigned_le(output, value);
}

void write_u64(std::ostream& output, std::uint64_t value) {
    write_unsigned_le(output, value);
}

void write_string(std::ostream& output, std::string_view value) {
    write_u64(output, value.size());
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_string_array(std::ostream& output,
                        std::initializer_list<std::string_view> values) {
    write_u32(output, 9);  // array
    write_u32(output, 8);  // string elements
    write_u64(output, values.size());
    for (std::string_view value : values) {
        write_string(output, value);
    }
}

[[nodiscard]] std::filesystem::path write_fixture() {
    const auto path = std::filesystem::temp_directory_path() /
                      "oracle-phase1c1-metadata-fixture.gguf";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create Phase 1C.1 fixture");
    }

    output.write("GGUF", 4);
    write_u32(output, 3);
    write_u64(output, 1);  // tensor count
    write_u64(output, 6);  // metadata count

    write_string(output, "general.architecture");
    write_u32(output, 8);
    write_string(output, "oracle-hybrid");

    write_string(output, "general.alignment");
    write_u32(output, 4);
    write_u32(output, 32);

    write_string(output, "model.context_length");
    write_u32(output, 4);
    write_u32(output, 32768);

    write_string(output, "tokenizer.ggml.tokens");
    write_string_array(output, {"hello", "world", "!"});

    write_string(output, "tokenizer.ggml.merges");
    write_string_array(output, {"h e", "he llo"});

    write_string(output, "tokenizer.chat_template");
    write_u32(output, 8);
    write_string(output, std::string(700, 'x'));

    write_string(output, "token_embd.weight");
    write_u32(output, 1);
    write_u64(output, 4);
    write_u32(output, 0);  // F32
    write_u64(output, 0);

    const std::uint64_t current = static_cast<std::uint64_t>(output.tellp());
    const std::uint64_t aligned = (current + 31ULL) & ~31ULL;
    for (std::uint64_t offset = current; offset < aligned; ++offset) {
        output.put('\0');
    }
    for (float value : {1.0F, 2.0F, 3.0F, 4.0F}) {
        write_u32(output, std::bit_cast<std::uint32_t>(value));
    }
    output.close();
    return path;
}

void test_metadata_export(TestRunner& runner) {
    const std::filesystem::path path = write_fixture();
    const oracle::model::GgufFile file = oracle::model::GgufReader::read(path);

    const auto* tokens = file.find_metadata("tokenizer.ggml.tokens");
    runner.expect(tokens != nullptr, "token metadata exists");
    runner.expect(tokens != nullptr &&
                      oracle::model::gguf_value_preview(tokens->value, 2).find(
                          "array<string>[3]") != std::string::npos,
                  "human preview reports array type and length");
    runner.expect(tokens != nullptr &&
                      oracle::model::gguf_value_preview(tokens->value, 2).find("...") !=
                          std::string::npos,
                  "human preview truncates large arrays");

    const oracle::model::GgufMetadataQuery tokenizer_query{
        .exact_key = {}, .prefix = "tokenizer.ggml."};
    const std::string tokenizer_json =
        oracle::model::gguf_metadata_report_json(file, tokenizer_query);
    runner.expect(tokenizer_json.find("\"selected_metadata_count\":2") !=
                      std::string::npos,
                  "prefix export reports selected metadata count");
    runner.expect(tokenizer_json.find("\"hello\"") != std::string::npos,
                  "full metadata export contains tokenizer values");
    runner.expect(tokenizer_json.find("general.architecture") == std::string::npos,
                  "prefix export excludes unrelated metadata");

    const oracle::model::GgufMetadataQuery exact_query{
        .exact_key = "general.architecture", .prefix = {}};
    const std::string exact_json =
        oracle::model::gguf_metadata_report_json(file, exact_query);
    runner.expect(exact_json.find("oracle-hybrid") != std::string::npos,
                  "exact-key export contains requested value");
    runner.expect(exact_json.find("tokenizer.ggml.tokens") == std::string::npos,
                  "exact-key export excludes other values");

    const std::string compact_tokens = oracle::model::gguf_metadata_entries_json(
        file,
        oracle::model::GgufMetadataJsonMode::compact,
        oracle::model::GgufMetadataQuery{.exact_key = "tokenizer.ggml.tokens"});
    runner.expect(compact_tokens.find("\"length\":3") != std::string::npos,
                  "compact array metadata reports length");
    runner.expect(compact_tokens.find("\"hello\"") == std::string::npos,
                  "compact array metadata omits full values");

    const std::string compact_template = oracle::model::gguf_metadata_entries_json(
        file,
        oracle::model::GgufMetadataJsonMode::compact,
        oracle::model::GgufMetadataQuery{.exact_key = "tokenizer.chat_template"});
    runner.expect(compact_template.find("\"length\":700") != std::string::npos,
                  "compact long-string metadata reports length");
    runner.expect(compact_template.find("\"preview\"") != std::string::npos,
                  "compact long-string metadata includes bounded preview");

    const std::string descriptor_summary = oracle::model::gguf_summary_json(file);
    runner.expect(descriptor_summary.find("\"metadata\":[") != std::string::npos,
                  "descriptor summary includes compact metadata entries");
    runner.expect(descriptor_summary.find("\"length\":3") != std::string::npos,
                  "descriptor summary keeps array shape information");

    oracle::model::MappedGgufModel mapped(path);
    const std::string mapped_summary = oracle::model::mapped_gguf_summary_json(mapped);
    runner.expect(mapped_summary.find("\"metadata\":[") != std::string::npos,
                  "verified mapping summary includes compact metadata entries");
    runner.expect(mapped_summary.find("oracle-hybrid") != std::string::npos,
                  "verified mapping summary includes scalar metadata values");

    runner.expect_throws(
        [&file] {
            static_cast<void>(oracle::model::gguf_metadata_entries_json(
                file,
                oracle::model::GgufMetadataJsonMode::full,
                oracle::model::GgufMetadataQuery{
                    .exact_key = "general.architecture", .prefix = "general."}));
        },
        "metadata export rejects exact-key and prefix combination");

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    TestRunner runner;
    try {
        test_metadata_export(runner);
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT TEST ERROR: " << error.what() << '\n';
        return 1;
    }

    if (runner.failures() == 0) {
        std::cout << "all Phase 1C.1 tests passed\n";
    }
    return runner.result();
}
