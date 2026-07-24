#include "oracle/core/mapped_file.hpp"
#include "oracle/model/ggml_type.hpp"
#include "oracle/model/mapped_gguf.hpp"

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
#include <span>
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

struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint32_t type{0};
    std::uint64_t relative_offset{0};
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

void write_string(std::ostream& output, std::string_view value) {
    write_u64(output, value.size());
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

[[nodiscard]] std::filesystem::path fixture_path(std::string_view name) {
    return std::filesystem::temp_directory_path() /
           ("oracle-phase1c-" + std::string(name) + ".gguf");
}

std::filesystem::path write_fixture(std::string_view label,
                                    std::span<const TensorSpec> tensors,
                                    std::uint32_t alignment,
                                    std::span<const std::byte> tensor_data) {
    const std::filesystem::path path = fixture_path(label);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create Phase 1C fixture");
    }

    output.write("GGUF", 4);
    write_u32(output, 3);
    write_u64(output, tensors.size());
    write_u64(output, 2);

    write_string(output, "general.architecture");
    write_u32(output, 8);
    write_string(output, "oracle-test");

    write_string(output, "general.alignment");
    write_u32(output, 4);
    write_u32(output, alignment);

    for (const TensorSpec& tensor : tensors) {
        write_string(output, tensor.name);
        write_u32(output, static_cast<std::uint32_t>(tensor.dimensions.size()));
        for (const std::uint64_t dimension : tensor.dimensions) {
            write_u64(output, dimension);
        }
        write_u32(output, tensor.type);
        write_u64(output, tensor.relative_offset);
    }

    const std::uint64_t current = static_cast<std::uint64_t>(output.tellp());
    const std::uint64_t aligned =
        (current + static_cast<std::uint64_t>(alignment - 1)) &
        ~static_cast<std::uint64_t>(alignment - 1);
    for (std::uint64_t offset = current; offset < aligned; ++offset) {
        output.put('\0');
    }
    output.write(reinterpret_cast<const char*>(tensor_data.data()),
                 static_cast<std::streamsize>(tensor_data.size()));
    output.close();
    return path;
}

[[nodiscard]] std::vector<std::byte> patterned_bytes(std::size_t size) {
    std::vector<std::byte> bytes(size);
    for (std::size_t index = 0; index < size; ++index) {
        bytes[index] = static_cast<std::byte>(index & 0xffU);
    }
    return bytes;
}

void test_type_layouts(TestRunner& runner) {
    const auto* f32 = oracle::model::ggml_type_layout(0);
    const auto* q4 = oracle::model::ggml_type_layout(2);
    const auto* q4k = oracle::model::ggml_type_layout(12);
    runner.expect(f32 != nullptr && f32->block_elements == 1 && f32->bytes_per_block == 4,
                  "F32 layout is registered");
    runner.expect(q4 != nullptr && q4->block_elements == 32 && q4->bytes_per_block == 18,
                  "Q4_0 layout is registered");
    runner.expect(q4k != nullptr && q4k->block_elements == 256 &&
                      q4k->bytes_per_block == 144,
                  "Q4_K layout is registered");
    runner.expect(oracle::model::ggml_type_layout(4) == nullptr,
                  "removed GGML types are rejected");
    runner.expect(oracle::model::ggml_type_layout(999) == nullptr,
                  "unknown GGML types are rejected");
    runner.expect(oracle::model::ggml_tensor_byte_size(
                      std::array<std::uint64_t, 2>{4, 3}, 0) == 48,
                  "F32 tensor byte size is calculated by rows");
    runner.expect(oracle::model::ggml_tensor_byte_size(
                      std::array<std::uint64_t, 2>{32, 2}, 2) == 36,
                  "quantized tensor byte size is calculated by blocks");
    runner.expect_throws(
        [] {
            static_cast<void>(oracle::model::ggml_tensor_byte_size(
                std::array<std::uint64_t, 1>{31}, 2));
        },
        "quantized row dimensions must match block size");
    runner.expect_throws(
        [] {
            static_cast<void>(oracle::model::ggml_tensor_element_count(
                std::array<std::uint64_t, 2>{
                    std::numeric_limits<std::uint64_t>::max(), 2}));
        },
        "tensor element count overflow is rejected");
}

void test_mapped_file_move(TestRunner& runner) {
    const std::vector<std::byte> data = patterned_bytes(16);
    const std::array<TensorSpec, 1> tensors{{{"mapped.test", {4}, 0, 0}}};
    const auto path = write_fixture("mapped-file", tensors, 32, data);

    oracle::core::MappedFile first(path);
    const std::size_t size = first.size();
    const std::byte first_byte = first.data()[0];
    oracle::core::MappedFile second(std::move(first));
    runner.expect(first.empty(), "moved-from mapped file is empty");
    runner.expect(second.size() == size && second.data()[0] == first_byte,
                  "mapped file move preserves mapping");

    std::filesystem::remove(path);
}

void test_valid_mapping(TestRunner& runner) {
    const std::array<TensorSpec, 2> tensors{{
        {"token_embd.weight", {4, 3}, 0, 0},
        {"blk.0.attn.weight", {32, 2}, 2, 64},
    }};
    const std::vector<std::byte> data = patterned_bytes(100);
    const auto path = write_fixture("valid", tensors, 64, data);

    oracle::model::MappedGgufModel model(path);
    runner.expect(model.stats().tensor_count == 2, "mapped model validates every tensor");
    runner.expect(model.stats().validated_payload_bytes == 84,
                  "mapped model reports validated payload bytes");
    runner.expect(model.stats().tensor_span_bytes == 100,
                  "mapped model reports tensor span including padding");

    const auto* embedding = model.find_tensor("token_embd.weight");
    const auto* quantized = model.find_tensor("blk.0.attn.weight");
    runner.expect(embedding != nullptr && embedding->bytes().size() == 48,
                  "F32 tensor view has correct byte size");
    runner.expect(quantized != nullptr && quantized->bytes().size() == 36,
                  "Q4_0 tensor view has correct byte size");
    runner.expect(embedding != nullptr &&
                      embedding->bytes().data() ==
                          model.mapping().data() + embedding->absolute_offset(),
                  "tensor view points directly into mapped model bytes");
    runner.expect(quantized != nullptr && quantized->layout().quantized,
                  "tensor view exposes GGML type layout");
    runner.expect(model.find_tensor("missing") == nullptr,
                  "tensor registry returns null for missing names");
    runner.expect_throws([&model] { static_cast<void>(model.tensor("missing")); },
                         "throwing tensor lookup reports missing names");
    runner.expect(oracle::model::mapped_gguf_summary_json(model).find(
                      "validated_payload_bytes") != std::string::npos,
                  "verified JSON includes mapping telemetry");

    const std::byte mapped_byte = embedding->bytes()[7];
    std::filesystem::remove(path);
    runner.expect(embedding->bytes()[7] == mapped_byte,
                  "POSIX mapping remains valid after the file is unlinked");
}

void test_truncated_payload(TestRunner& runner) {
    const std::array<TensorSpec, 1> tensors{{{"truncated", {4, 3}, 0, 0}}};
    const std::vector<std::byte> data = patterned_bytes(12);
    const auto path = write_fixture("truncated", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "truncated tensor payload is rejected");
    std::filesystem::remove(path);
}

void test_invalid_block_shape(TestRunner& runner) {
    const std::array<TensorSpec, 1> tensors{{{"bad.block", {31}, 2, 0}}};
    const std::vector<std::byte> data = patterned_bytes(64);
    const auto path = write_fixture("bad-block", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "invalid quantized row shape is rejected");
    std::filesystem::remove(path);
}


void test_misaligned_offset(TestRunner& runner) {
    const std::array<TensorSpec, 1> tensors{{{"misaligned", {4}, 0, 16}}};
    const std::vector<std::byte> data = patterned_bytes(64);
    const auto path = write_fixture("misaligned", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "misaligned tensor offsets are rejected");
    std::filesystem::remove(path);
}

void test_overlapping_payloads(TestRunner& runner) {
    const std::array<TensorSpec, 2> tensors{{
        {"first", {16}, 0, 0},
        {"second", {8}, 0, 32},
    }};
    const std::vector<std::byte> data = patterned_bytes(96);
    const auto path = write_fixture("overlap", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "overlapping tensor payloads are rejected");
    std::filesystem::remove(path);
}

void test_duplicate_names(TestRunner& runner) {
    const std::array<TensorSpec, 2> tensors{{
        {"duplicate", {4}, 0, 0},
        {"duplicate", {4}, 0, 32},
    }};
    const std::vector<std::byte> data = patterned_bytes(48);
    const auto path = write_fixture("duplicate", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "duplicate tensor names are rejected");
    std::filesystem::remove(path);
}

void test_unsupported_type(TestRunner& runner) {
    const std::array<TensorSpec, 1> tensors{{{"unsupported", {4}, 999, 0}}};
    const std::vector<std::byte> data = patterned_bytes(32);
    const auto path = write_fixture("unsupported", tensors, 32, data);
    runner.expect_throws([&path] { oracle::model::MappedGgufModel model(path); },
                         "unsupported GGML types are rejected");
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    TestRunner runner;
    try {
        test_type_layouts(runner);
        test_mapped_file_move(runner);
        test_valid_mapping(runner);
        test_truncated_payload(runner);
        test_invalid_block_shape(runner);
        test_misaligned_offset(runner);
        test_overlapping_payloads(runner);
        test_duplicate_names(runner);
        test_unsupported_type(runner);
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT TEST ERROR: " << error.what() << '\n';
        return 1;
    }

    if (runner.failures() == 0) {
        std::cout << "all Phase 1C tests passed\n";
    }
    return runner.result();
}
