#include "oracle/backend/cpu/quantized_reference.hpp"
#include "oracle/model/storage_decode.hpp"
#include "oracle/model/ggml_type.hpp"
#include "oracle/model/mapped_gguf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void store_u16_le(std::span<std::byte> bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset] = static_cast<std::byte>(value & 0xFFU);
    bytes[offset + 1] = static_cast<std::byte>(value >> 8U);
}

[[nodiscard]] std::array<std::byte, oracle::model::q5_k_block_bytes> q5_fixture() {
    std::array<std::byte, oracle::model::q5_k_block_bytes> block{};
    store_u16_le(block, 0, 0x3800U);  // d = 0.5
    store_u16_le(block, 2, 0x3400U);  // dmin = 0.25
    constexpr std::array<std::uint8_t, 12> scales{
        2, 2, 2, 2, 3, 3, 3, 3, 0x32, 0x32, 0x32, 0x32};
    for (std::size_t index = 0; index < scales.size(); ++index) {
        block[4 + index] = static_cast<std::byte>(scales[index]);
    }
    std::fill(block.begin() + 16, block.begin() + 48, std::byte{0xFF});
    std::fill(block.begin() + 48, block.end(), std::byte{0x21});
    return block;
}

[[nodiscard]] std::array<std::byte, oracle::model::q6_k_block_bytes> q6_fixture() {
    std::array<std::byte, oracle::model::q6_k_block_bytes> block{};
    std::fill(block.begin(), block.begin() + 128, std::byte{0x21});
    std::fill(block.begin() + 128, block.begin() + 192, std::byte{0xFF});
    std::fill(block.begin() + 192, block.begin() + 208, std::byte{0x02});
    store_u16_le(block, 208, 0x3800U);  // d = 0.5
    return block;
}

// Phase 2F Slice 1B: block_q8_0 layout is { ggml_half d; int8_t qs[32]; } (34 bytes,
// authoritative source: beellama ggml/src/ggml-common.h at commit a620cbd48). d=1.0
// (0x3C00) makes decoded[i] == qs[i] exactly (int8 -> float is exact in this range),
// so this fixture doubles as a hand-verifiable known-answer row. Deliberately covers
// zero, +1/-1, the int8 extremes (+127/-128), and a symmetric +/-N ramp.
[[nodiscard]] std::array<std::byte, oracle::model::q8_0_block_bytes> q8_0_fixture() {
    std::array<std::byte, oracle::model::q8_0_block_bytes> block{};
    store_u16_le(block, 0, 0x3C00U);  // d = 1.0
    constexpr std::array<std::int8_t, 32> quants{
        0, 1, -1, 127, -128, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7,
        -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15};
    for (std::size_t index = 0; index < quants.size(); ++index) {
        block[2 + index] = static_cast<std::byte>(static_cast<std::uint8_t>(quants[index]));
    }
    return block;
}

void test_half_conversions() {
    using oracle::model::bf16_to_f32;
    using oracle::model::f16_to_f32;

    require_near(f16_to_f32(0x3C00U), 1.0F, 0.0F, "F16 one mismatch");
    require_near(f16_to_f32(0xC000U), -2.0F, 0.0F, "F16 negative mismatch");
    require_near(f16_to_f32(0x0001U), std::ldexp(1.0F, -24), 0.0F,
                 "F16 subnormal mismatch");
    require(std::isinf(f16_to_f32(0x7C00U)), "F16 infinity mismatch");
    require(std::isnan(f16_to_f32(0x7E00U)), "F16 NaN mismatch");
    require(std::signbit(f16_to_f32(0x8000U)), "F16 negative zero sign mismatch");

    require_near(bf16_to_f32(0x3F80U), 1.0F, 0.0F, "BF16 one mismatch");
    require_near(bf16_to_f32(0xC000U), -2.0F, 0.0F, "BF16 negative mismatch");
    require(std::isinf(bf16_to_f32(0x7F80U)), "BF16 infinity mismatch");
    require(std::isnan(bf16_to_f32(0x7FC1U)), "BF16 NaN mismatch");
}

void test_plain_rows() {
    std::array<std::byte, 8> f16_bytes{};
    constexpr std::array<std::uint16_t, 4> f16_bits{0x3C00U, 0xC000U, 0x4200U, 0x0000U};
    for (std::size_t index = 0; index < f16_bits.size(); ++index) {
        store_u16_le(f16_bytes, index * 2, f16_bits[index]);
    }
    std::array<float, 4> decoded{};
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_f16, 4, f16_bytes),
        decoded);
    constexpr std::array<float, 4> expected{1.0F, -2.0F, 3.0F, 0.0F};
    require(decoded == expected, "F16 row decode mismatch");

    std::array<std::byte, 8> bf16_bytes{};
    constexpr std::array<std::uint16_t, 4> bf16_bits{0x3F80U, 0xC000U, 0x4040U, 0x0000U};
    for (std::size_t index = 0; index < bf16_bits.size(); ++index) {
        store_u16_le(bf16_bytes, index * 2, bf16_bits[index]);
    }
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_bf16, 4, bf16_bytes),
        decoded);
    require(decoded == expected, "BF16 row decode mismatch");
}

void test_q5_k_known_answer() {
    const auto block = q5_fixture();
    std::array<float, oracle::model::qk_k> decoded{};
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_q5_k,
                                             oracle::model::qk_k,
                                             block),
        decoded);
    for (std::size_t group = 0; group < 4; ++group) {
        for (std::size_t lane = 0; lane < 32; ++lane) {
            require_near(decoded[group * 64 + lane], 16.25F, 0.0F,
                         "Q5_K low-half known answer mismatch");
            require_near(decoded[group * 64 + 32 + lane], 17.25F, 0.0F,
                         "Q5_K high-half known answer mismatch");
        }
    }
}

void test_q6_k_known_answer() {
    const auto block = q6_fixture();
    std::array<float, oracle::model::qk_k> decoded{};
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_q6_k,
                                             oracle::model::qk_k,
                                             block),
        decoded);
    for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t base = half * 128;
        for (std::size_t lane = 0; lane < 64; ++lane) {
            require_near(decoded[base + lane], 17.0F, 0.0F,
                         "Q6_K lower segments known answer mismatch");
            require_near(decoded[base + 64 + lane], 18.0F, 0.0F,
                         "Q6_K upper segments known answer mismatch");
        }
    }
}

// Item 1 (geometry) + item 2 (known-answer) + item 3 (signed quant values, incl. the
// int8 extremes) from the Phase 2F Slice 1B taskblock.
void test_q8_0_known_answer() {
    const auto* layout = oracle::model::ggml_type_layout(oracle::model::ggml_type_q8_0);
    require(layout != nullptr, "Q8_0 layout must be registered");
    require(layout->block_elements == oracle::model::q8_0_block_elements,
            "Q8_0 block element count mismatch");
    require(layout->bytes_per_block == oracle::model::q8_0_block_bytes,
            "Q8_0 block byte size mismatch");
    require(layout->quantized, "Q8_0 must be marked quantized");

    const auto block = q8_0_fixture();
    std::array<float, oracle::model::q8_0_block_elements> decoded{};
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_q8_0,
                                             oracle::model::q8_0_block_elements,
                                             block),
        decoded);
    constexpr std::array<float, 32> expected{
        0, 1, -1, 127, -128, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7,
        -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_near(decoded[index], expected[index], 0.0F,
                     "Q8_0 known-answer mismatch at index " + std::to_string(index));
    }
}

// Item 4: multi-block row, two blocks with distinct deltas and distinct quant patterns.
void test_q8_0_multi_block_row() {
    const auto first = q8_0_fixture();
    std::array<std::byte, oracle::model::q8_0_block_bytes> second{};
    store_u16_le(second, 0, 0x3800U);  // d = 0.5
    for (std::size_t index = 0; index < oracle::model::q8_0_block_elements; ++index) {
        second[2 + index] = static_cast<std::byte>(static_cast<std::uint8_t>(10));
    }

    std::array<std::byte, oracle::model::q8_0_block_bytes * 2> two_blocks{};
    std::copy(first.begin(), first.end(), two_blocks.begin());
    std::copy(second.begin(), second.end(),
              two_blocks.begin() + oracle::model::q8_0_block_bytes);

    std::array<float, oracle::model::q8_0_block_elements * 2> decoded{};
    oracle::model::decode_storage_row(
        oracle::model::make_storage_row_view(oracle::model::ggml_type_q8_0,
                                             oracle::model::q8_0_block_elements * 2,
                                             two_blocks),
        decoded);
    require_near(decoded[3], 127.0F, 0.0F, "Q8_0 multi-block first-block value mismatch");
    require_near(decoded[oracle::model::q8_0_block_elements], 5.0F, 0.0F,
                 "Q8_0 multi-block second-block value mismatch (10 * 0.5)");
    require_near(decoded[oracle::model::q8_0_block_elements + 31], 5.0F, 0.0F,
                 "Q8_0 multi-block second-block last-lane value mismatch");
}

void test_reference_dot_and_matvec() {
    const auto q5 = q5_fixture();
    std::array<float, oracle::model::qk_k> ones{};
    ones.fill(1.0F);
    const auto row = oracle::model::make_storage_row_view(oracle::model::ggml_type_q5_k,
                                                          oracle::model::qk_k,
                                                          q5);
    require_near(oracle::backend::cpu::reference_storage_dot(row, ones),
                 4288.0F,
                 0.0F,
                 "Q5_K reference dot mismatch");

    std::array<std::byte, oracle::model::q5_k_block_bytes * 2> two_blocks{};
    std::copy(q5.begin(), q5.end(), two_blocks.begin());
    std::copy(q5.begin(), q5.end(),
              two_blocks.begin() + oracle::model::q5_k_block_bytes);
    std::array<float, oracle::model::qk_k * 2> two_block_vector{};
    two_block_vector.fill(1.0F);
    require_near(oracle::backend::cpu::reference_storage_dot(
                     oracle::model::make_storage_row_view(oracle::model::ggml_type_q5_k,
                                                          oracle::model::qk_k * 2,
                                                          two_blocks),
                     two_block_vector),
                 8576.0F,
                 0.0F,
                 "Q5_K multi-block reference dot mismatch");

    std::array<std::byte, oracle::model::q5_k_block_bytes * 2> matrix{};
    std::copy(q5.begin(), q5.end(), matrix.begin());
    auto second = q5;
    std::fill(second.begin() + 16, second.end(), std::byte{0x00});
    std::copy(second.begin(), second.end(), matrix.begin() + oracle::model::q5_k_block_bytes);
    std::array<float, 2> output{};
    oracle::backend::cpu::reference_storage_matvec(oracle::model::ggml_type_q5_k,
                                                    2,
                                                    oracle::model::qk_k,
                                                    matrix,
                                                    ones,
                                                    output);
    require_near(output[0], 4288.0F, 0.0F, "Q5_K matvec first row mismatch");
    require_near(output[1], -192.0F, 0.0F, "Q5_K matvec second row mismatch");

    const auto q6 = q6_fixture();
    require_near(oracle::backend::cpu::reference_storage_dot(
                     oracle::model::make_storage_row_view(oracle::model::ggml_type_q6_k,
                                                          oracle::model::qk_k,
                                                          q6),
                     ones),
                 4480.0F,
                 0.0F,
                 "Q6_K reference dot mismatch");

    // Item 7: Q8_0 scalar dot. sum(quants) in q8_0_fixture() == 14 (the +1/-1, +/-2..14
    // pairs cancel; only the unpaired 0, the +127/-128 pair (-1), and the trailing +15
    // survive: 0 + (-1) + 15 == 14), times d == 1.0.
    const auto q8 = q8_0_fixture();
    std::array<float, oracle::model::q8_0_block_elements> ones_q8{};
    ones_q8.fill(1.0F);
    require_near(oracle::backend::cpu::reference_storage_dot(
                     oracle::model::make_storage_row_view(oracle::model::ggml_type_q8_0,
                                                          oracle::model::q8_0_block_elements,
                                                          q8),
                     ones_q8),
                 14.0F,
                 0.0F,
                 "Q8_0 reference dot mismatch");
}


void test_mapped_tensor_row_adapter() {
    const auto q5 = q5_fixture();
    std::array<std::byte, oracle::model::q5_k_block_bytes * 2> matrix{};
    std::copy(q5.begin(), q5.end(), matrix.begin());
    auto second = q5;
    std::fill(second.begin() + 16, second.end(), std::byte{0x00});
    std::copy(second.begin(), second.end(),
              matrix.begin() + oracle::model::q5_k_block_bytes);

    oracle::model::GgufTensorInfo info{
        "adapter.weight", {oracle::model::qk_k, 2}, oracle::model::ggml_type_q5_k, 0};
    const auto* layout = oracle::model::ggml_type_layout(oracle::model::ggml_type_q5_k);
    require(layout != nullptr, "Q5_K layout must exist for mapped adapter test");
    const oracle::model::GgufTensorView tensor(
        &info, layout, matrix.data(), matrix.size(), 4096);

    require(oracle::model::gguf_tensor_row_count(tensor) == 2,
            "mapped tensor row count mismatch");
    const auto first_row = oracle::model::make_storage_row_view(tensor, 0);
    const auto second_row = oracle::model::make_storage_row_view(tensor, 1);
    require(first_row.bytes.data() == tensor.bytes().data(),
            "first mapped row must begin at tensor storage");
    require(second_row.bytes.data() ==
                tensor.bytes().data() + oracle::model::q5_k_block_bytes,
            "second mapped row must advance by exact row bytes");

    std::array<float, oracle::model::qk_k> ones{};
    ones.fill(1.0F);
    require_near(oracle::backend::cpu::reference_storage_dot(first_row, ones),
                 4288.0F,
                 0.0F,
                 "mapped first-row dot mismatch");
    require_near(oracle::backend::cpu::reference_storage_dot(second_row, ones),
                 -192.0F,
                 0.0F,
                 "mapped second-row dot mismatch");

    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(tensor, 2)); },
        "mapped tensor adapter must reject an out-of-range row");

    const oracle::model::GgufTensorView malformed(
        &info, layout, matrix.data(), matrix.size() - 1, 4096);
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(malformed, 0)); },
        "mapped tensor adapter must reject inconsistent tensor byte geometry");
}

// Item 8: Q8_0 mapped-row adapter, same bounds/geometry discipline as the Q5_K adapter
// test above -- no copies, exact row-byte offsets, out-of-range/malformed rejection.
void test_q8_0_mapped_tensor_row_adapter() {
    const auto first = q8_0_fixture();
    std::array<std::byte, oracle::model::q8_0_block_bytes> second{};
    store_u16_le(second, 0, 0x3800U);  // d = 0.5
    for (std::size_t index = 0; index < oracle::model::q8_0_block_elements; ++index) {
        second[2 + index] = static_cast<std::byte>(static_cast<std::uint8_t>(4));
    }

    std::array<std::byte, oracle::model::q8_0_block_bytes * 2> matrix{};
    std::copy(first.begin(), first.end(), matrix.begin());
    std::copy(second.begin(), second.end(), matrix.begin() + oracle::model::q8_0_block_bytes);

    oracle::model::GgufTensorInfo info{"adapter.q8_0.weight",
                                       {oracle::model::q8_0_block_elements, 2},
                                       oracle::model::ggml_type_q8_0,
                                       0};
    const auto* layout = oracle::model::ggml_type_layout(oracle::model::ggml_type_q8_0);
    require(layout != nullptr, "Q8_0 layout must exist for mapped adapter test");
    const oracle::model::GgufTensorView tensor(&info, layout, matrix.data(), matrix.size(), 8192);

    require(oracle::model::gguf_tensor_row_count(tensor) == 2,
            "Q8_0 mapped tensor row count mismatch");
    const auto first_row = oracle::model::make_storage_row_view(tensor, 0);
    const auto second_row = oracle::model::make_storage_row_view(tensor, 1);
    require(first_row.bytes.data() == tensor.bytes().data(),
            "Q8_0 first mapped row must begin at tensor storage");
    require(second_row.bytes.data() ==
                tensor.bytes().data() + oracle::model::q8_0_block_bytes,
            "Q8_0 second mapped row must advance by exact row bytes (no copy)");

    std::array<float, oracle::model::q8_0_block_elements> ones{};
    ones.fill(1.0F);
    require_near(oracle::backend::cpu::reference_storage_dot(first_row, ones),
                 14.0F,
                 0.0F,
                 "Q8_0 mapped first-row dot mismatch");
    require_near(oracle::backend::cpu::reference_storage_dot(second_row, ones),
                 64.0F,
                 0.0F,
                 "Q8_0 mapped second-row dot mismatch (32 * 4 * 0.5)");

    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(tensor, 2)); },
        "Q8_0 mapped tensor adapter must reject an out-of-range row");

    const oracle::model::GgufTensorView malformed(&info, layout, matrix.data(),
                                                  matrix.size() - 1, 8192);
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(malformed, 0)); },
        "Q8_0 mapped tensor adapter must reject inconsistent tensor byte geometry");
}

void test_validation() {
    std::array<std::byte, 2> bytes{};
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(
                  oracle::model::ggml_type_q5_k, oracle::model::qk_k, bytes)); },
        "Q5_K row must reject truncated storage");
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(0, 1, bytes)); },
        "unsupported storage decoder type must be rejected");

    const auto q5 = q5_fixture();
    std::array<float, 1> short_output{};
    require_throws(
        [&] {
            oracle::model::decode_storage_row(
                oracle::model::make_storage_row_view(oracle::model::ggml_type_q5_k,
                                                     oracle::model::qk_k,
                                                     q5),
                short_output);
        },
        "row decode must reject output length mismatch");

    // Item 5: Q8_0 truncated-row rejection (one byte short of a full 34-byte block).
    std::array<std::byte, oracle::model::q8_0_block_bytes - 1> truncated_q8{};
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(
                  oracle::model::ggml_type_q8_0, oracle::model::q8_0_block_elements,
                  truncated_q8)); },
        "Q8_0 row must reject truncated storage");

    // Item 6: Q8_0 malformed row-width rejection (17 is not divisible by the 32-element
    // block size).
    const auto q8 = q8_0_fixture();
    require_throws(
        [&] { static_cast<void>(oracle::model::make_storage_row_view(
                  oracle::model::ggml_type_q8_0, 17, q8)); },
        "not divisible by block size");
}

}  // namespace

int main() {
    try {
        test_half_conversions();
        test_plain_rows();
        test_q5_k_known_answer();
        test_q6_k_known_answer();
        test_q8_0_known_answer();
        test_q8_0_multi_block_row();
        test_reference_dot_and_matvec();
        test_mapped_tensor_row_adapter();
        test_q8_0_mapped_tensor_row_adapter();
        test_validation();
        std::cout << "Phase 2A storage decoding tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2A test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
