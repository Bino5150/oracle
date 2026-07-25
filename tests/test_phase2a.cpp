#include "oracle/backend/cpu/quantized_reference.hpp"
#include "oracle/model/storage_decode.hpp"

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
}

}  // namespace

int main() {
    try {
        test_half_conversions();
        test_plain_rows();
        test_q5_k_known_answer();
        test_q6_k_known_answer();
        test_reference_dot_and_matvec();
        test_validation();
        std::cout << "Phase 2A storage decoding tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "Phase 2A test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
