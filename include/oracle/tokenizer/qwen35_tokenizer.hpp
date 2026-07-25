#pragma once

#include "oracle/model/gguf.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace oracle::tokenizer {

using TokenId = std::uint32_t;

enum class TokenType : std::uint32_t {
    undefined = 0,
    normal = 1,
    unknown = 2,
    control = 3,
    user_defined = 4,
    unused = 5,
    byte = 6,
};

struct EncodeOptions {
    bool parse_special_tokens{false};
};

struct DecodeOptions {
    bool skip_special_tokens{false};
};

class Qwen35Tokenizer {
public:
    explicit Qwen35Tokenizer(const model::GgufFile& file);

    [[nodiscard]] std::vector<TokenId> encode(
        std::string_view text,
        EncodeOptions options = {}) const;
    [[nodiscard]] std::string decode(
        std::span<const TokenId> token_ids,
        DecodeOptions options = {}) const;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t merge_count() const noexcept;
    [[nodiscard]] std::string_view token_text(TokenId token_id) const;
    [[nodiscard]] TokenType token_type(TokenId token_id) const;
    [[nodiscard]] std::optional<TokenId> find_token(std::string_view token) const noexcept;
    [[nodiscard]] bool is_special(TokenId token_id) const;

    [[nodiscard]] std::optional<TokenId> bos_token_id() const noexcept;
    [[nodiscard]] std::optional<TokenId> eos_token_id() const noexcept;
    [[nodiscard]] std::optional<TokenId> padding_token_id() const noexcept;
    [[nodiscard]] std::string_view chat_template() const noexcept;

private:
    struct PairHash {
        [[nodiscard]] std::size_t operator()(
            const std::pair<std::string, std::string>& value) const noexcept;
    };

    [[nodiscard]] std::vector<TokenId> encode_normal(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> pretokenize(std::string_view text) const;
    [[nodiscard]] std::vector<std::string> apply_bpe(std::string encoded_piece) const;
    [[nodiscard]] std::string byte_encode(std::string_view bytes) const;
    [[nodiscard]] std::string byte_decode(std::string_view encoded) const;

    std::vector<std::string> tokens_;
    std::vector<TokenType> token_types_;
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, std::uint32_t, PairHash>
        merge_ranks_;
    std::vector<std::pair<std::string, TokenId>> special_tokens_;
    std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte_;
    std::vector<std::string> byte_to_unicode_;

    std::optional<TokenId> bos_token_id_;
    std::optional<TokenId> eos_token_id_;
    std::optional<TokenId> padding_token_id_;
    std::string chat_template_;
};

[[nodiscard]] std::string token_ids_json(std::span<const TokenId> token_ids);

}  // namespace oracle::tokenizer
