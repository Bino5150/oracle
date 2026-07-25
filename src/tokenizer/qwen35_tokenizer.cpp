#include "oracle/tokenizer/qwen35_tokenizer.hpp"

#include "unicode_categories.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace oracle::tokenizer {
namespace {

struct Utf8Unit {
    std::uint32_t codepoint{0};
    std::size_t begin{0};
    std::size_t end{0};
    bool valid{false};
};

[[nodiscard]] std::string utf8_from_codepoint(std::uint32_t codepoint) {
    std::string output;
    if (codepoint <= 0x7FU) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
        output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
        output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0x10FFFFU) {
        output.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        throw std::invalid_argument("invalid Unicode code point");
    }
    return output;
}

[[nodiscard]] std::vector<Utf8Unit> decode_utf8_units(std::string_view text) {
    std::vector<Utf8Unit> units;
    units.reserve(text.size());
    std::size_t index = 0;
    while (index < text.size()) {
        const auto first = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        std::uint32_t minimum = 0;
        if (first < 0x80U) {
            length = 1;
            codepoint = first;
        } else if ((first & 0xE0U) == 0xC0U) {
            length = 2;
            codepoint = first & 0x1FU;
            minimum = 0x80U;
        } else if ((first & 0xF0U) == 0xE0U) {
            length = 3;
            codepoint = first & 0x0FU;
            minimum = 0x800U;
        } else if ((first & 0xF8U) == 0xF0U) {
            length = 4;
            codepoint = first & 0x07U;
            minimum = 0x10000U;
        }

        bool valid = length != 0 && index + length <= text.size();
        if (valid && length > 1) {
            for (std::size_t offset = 1; offset < length; ++offset) {
                const auto continuation =
                    static_cast<unsigned char>(text[index + offset]);
                if ((continuation & 0xC0U) != 0x80U) {
                    valid = false;
                    break;
                }
                codepoint = (codepoint << 6U) | (continuation & 0x3FU);
            }
            if (codepoint < minimum || codepoint > 0x10FFFFU ||
                (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
                valid = false;
            }
        }

        if (!valid) {
            units.push_back(Utf8Unit{0x110000U + first, index, index + 1, false});
            ++index;
        } else {
            units.push_back(Utf8Unit{codepoint, index, index + length, true});
            index += length;
        }
    }
    return units;
}

[[nodiscard]] bool is_letter_or_mark(const Utf8Unit& unit) noexcept {
    return unit.valid &&
           (detail::is_letter(unit.codepoint) || detail::is_mark(unit.codepoint));
}

[[nodiscard]] bool is_letter(const Utf8Unit& unit) noexcept {
    return unit.valid && detail::is_letter(unit.codepoint);
}

[[nodiscard]] bool is_number(const Utf8Unit& unit) noexcept {
    return unit.valid && detail::is_number(unit.codepoint);
}

[[nodiscard]] bool is_whitespace(const Utf8Unit& unit) noexcept {
    return unit.valid && detail::is_whitespace(unit.codepoint);
}

[[nodiscard]] bool is_newline(const Utf8Unit& unit) noexcept {
    return unit.valid && (unit.codepoint == '\r' || unit.codepoint == '\n');
}

[[nodiscard]] bool is_ascii_space(const Utf8Unit& unit) noexcept {
    return unit.valid && unit.codepoint == 0x20U;
}

[[nodiscard]] bool is_punctuation_class(const Utf8Unit& unit) noexcept {
    return !is_whitespace(unit) && !is_letter_or_mark(unit) && !is_number(unit);
}

[[nodiscard]] std::size_t contraction_length(std::string_view text,
                                             std::size_t byte_index) noexcept {
    if (byte_index >= text.size() || text[byte_index] != '\'') {
        return 0;
    }
    const auto matches = [&text, byte_index](std::string_view suffix) {
        if (byte_index + 1 + suffix.size() > text.size()) {
            return false;
        }
        for (std::size_t index = 0; index < suffix.size(); ++index) {
            const unsigned char actual =
                static_cast<unsigned char>(text[byte_index + 1 + index]);
            const unsigned char expected = static_cast<unsigned char>(suffix[index]);
            if (std::tolower(actual) != std::tolower(expected)) {
                return false;
            }
        }
        return true;
    };
    for (std::string_view suffix : {"re", "ve", "ll", "s", "t", "m", "d"}) {
        if (matches(suffix)) {
            return suffix.size() + 1;
        }
    }
    return 0;
}

[[nodiscard]] const model::GgufValue& require_metadata(const model::GgufFile& file,
                                                       std::string_view key) {
    const model::GgufMetadataEntry* entry = file.find_metadata(key);
    if (entry == nullptr) {
        throw std::runtime_error("missing required tokenizer metadata: " +
                                 std::string(key));
    }
    return entry->value;
}

[[nodiscard]] std::string require_string(const model::GgufFile& file,
                                         std::string_view key) {
    const auto* value = require_metadata(file, key).get_if<std::string>();
    if (value == nullptr) {
        throw std::runtime_error("tokenizer metadata must be string: " +
                                 std::string(key));
    }
    return *value;
}

[[nodiscard]] std::optional<std::uint32_t> optional_u32(
    const model::GgufFile& file,
    std::string_view key) {
    const model::GgufMetadataEntry* entry = file.find_metadata(key);
    if (entry == nullptr) {
        return std::nullopt;
    }
    const auto* value = entry->value.get_if<std::uint32_t>();
    if (value == nullptr) {
        throw std::runtime_error("tokenizer metadata must be uint32: " +
                                 std::string(key));
    }
    return *value;
}

[[nodiscard]] const model::GgufArray& require_array(
    const model::GgufFile& file,
    std::string_view key,
    model::GgufMetadataType element_type) {
    const auto* pointer =
        require_metadata(file, key).get_if<std::shared_ptr<model::GgufArray>>();
    if (pointer == nullptr || *pointer == nullptr) {
        throw std::runtime_error("tokenizer metadata must be array: " +
                                 std::string(key));
    }
    if ((*pointer)->element_type != element_type) {
        throw std::runtime_error("tokenizer metadata array has wrong element type: " +
                                 std::string(key));
    }
    return **pointer;
}


}  // namespace

std::size_t Qwen35Tokenizer::PairHash::operator()(
    const std::pair<std::string, std::string>& value) const noexcept {
    const std::size_t left = std::hash<std::string>{}(value.first);
    const std::size_t right = std::hash<std::string>{}(value.second);
    return left ^ (right + 0x9e3779b97f4a7c15ULL + (left << 6U) + (left >> 2U));
}

Qwen35Tokenizer::Qwen35Tokenizer(const model::GgufFile& file) {
    if (require_string(file, "tokenizer.ggml.model") != "gpt2") {
        throw std::runtime_error("Qwen35 tokenizer requires tokenizer.ggml.model=gpt2");
    }
    if (require_string(file, "tokenizer.ggml.pre") != "qwen35") {
        throw std::runtime_error("Qwen35 tokenizer requires tokenizer.ggml.pre=qwen35");
    }

    const model::GgufArray& token_array = require_array(
        file, "tokenizer.ggml.tokens", model::GgufMetadataType::string);
    const model::GgufArray& type_array = require_array(
        file, "tokenizer.ggml.token_type", model::GgufMetadataType::int32);
    const model::GgufArray& merge_array = require_array(
        file, "tokenizer.ggml.merges", model::GgufMetadataType::string);
    if (token_array.values.empty() || token_array.values.size() != type_array.values.size()) {
        throw std::runtime_error(
            "tokenizer tokens and token types must be non-empty and equal length");
    }
    if (token_array.values.size() > std::numeric_limits<TokenId>::max()) {
        throw std::runtime_error("tokenizer vocabulary exceeds TokenId range");
    }

    tokens_.reserve(token_array.values.size());
    token_types_.reserve(type_array.values.size());
    token_to_id_.reserve(token_array.values.size());
    for (std::size_t index = 0; index < token_array.values.size(); ++index) {
        const auto* token = token_array.values[index].get_if<std::string>();
        const auto* raw_type = type_array.values[index].get_if<std::int32_t>();
        if (token == nullptr || raw_type == nullptr || *raw_type < 0 || *raw_type > 6) {
            throw std::runtime_error("invalid tokenizer token or token type entry");
        }
        const TokenId token_id = static_cast<TokenId>(index);
        const auto [iterator, inserted] = token_to_id_.emplace(*token, token_id);
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error("duplicate tokenizer token: " + *token);
        }
        tokens_.push_back(*token);
        token_types_.push_back(static_cast<TokenType>(*raw_type));
        if (token_types_.back() == TokenType::control ||
            token_types_.back() == TokenType::user_defined) {
            special_tokens_.emplace_back(*token, token_id);
        }
    }
    std::ranges::sort(special_tokens_, [](const auto& left, const auto& right) {
        if (left.first.size() != right.first.size()) {
            return left.first.size() > right.first.size();
        }
        return left.second < right.second;
    });

    merge_ranks_.reserve(merge_array.values.size());
    for (std::size_t index = 0; index < merge_array.values.size(); ++index) {
        const auto* merge = merge_array.values[index].get_if<std::string>();
        if (merge == nullptr) {
            throw std::runtime_error("tokenizer merge entry must be string");
        }
        const std::size_t separator = merge->find(' ');
        if (separator == std::string::npos || separator == 0 ||
            separator + 1 >= merge->size() || merge->find(' ', separator + 1) != std::string::npos) {
            throw std::runtime_error("invalid tokenizer BPE merge: " + *merge);
        }
        std::pair<std::string, std::string> pair{
            merge->substr(0, separator), merge->substr(separator + 1)};
        if (!token_to_id_.contains(pair.first + pair.second)) {
            throw std::runtime_error("BPE merge result is absent from vocabulary: " + *merge);
        }
        const auto [iterator, inserted] =
            merge_ranks_.emplace(std::move(pair), static_cast<std::uint32_t>(index));
        static_cast<void>(iterator);
        if (!inserted) {
            throw std::runtime_error("duplicate tokenizer BPE merge: " + *merge);
        }
    }

    byte_to_unicode_.resize(256);
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
        const std::uint32_t codepoint = direct[byte] ? byte : 256U + extension++;
        byte_to_unicode_[byte] = utf8_from_codepoint(codepoint);
        unicode_to_byte_.emplace(codepoint, static_cast<std::uint8_t>(byte));
        if (!token_to_id_.contains(byte_to_unicode_[byte])) {
            throw std::runtime_error(
                "tokenizer vocabulary is missing a required byte token");
        }
    }

    bos_token_id_ = optional_u32(file, "tokenizer.ggml.bos_token_id");
    eos_token_id_ = optional_u32(file, "tokenizer.ggml.eos_token_id");
    padding_token_id_ = optional_u32(file, "tokenizer.ggml.padding_token_id");
    for (const auto id : {bos_token_id_, eos_token_id_, padding_token_id_}) {
        if (id && *id >= tokens_.size()) {
            throw std::runtime_error("tokenizer special token id is outside vocabulary");
        }
    }
    const model::GgufMetadataEntry* chat_template =
        file.find_metadata("tokenizer.chat_template");
    if (chat_template != nullptr) {
        const auto* value = chat_template->value.get_if<std::string>();
        if (value == nullptr) {
            throw std::runtime_error("tokenizer.chat_template must be string");
        }
        chat_template_ = *value;
    }
}

std::vector<TokenId> Qwen35Tokenizer::encode(std::string_view text,
                                             EncodeOptions options) const {
    if (!options.parse_special_tokens || special_tokens_.empty()) {
        return encode_normal(text);
    }

    std::vector<TokenId> output;
    std::size_t position = 0;
    while (position < text.size()) {
        std::size_t next_position = text.size();
        const std::pair<std::string, TokenId>* next_special = nullptr;
        for (const auto& special : special_tokens_) {
            const std::size_t found = text.find(special.first, position);
            if (found < next_position ||
                (found == next_position && next_special != nullptr &&
                 special.first.size() > next_special->first.size())) {
                next_position = found;
                next_special = &special;
            }
        }
        if (next_special == nullptr) {
            std::vector<TokenId> suffix = encode_normal(text.substr(position));
            output.insert(output.end(), suffix.begin(), suffix.end());
            break;
        }
        if (next_position > position) {
            std::vector<TokenId> prefix =
                encode_normal(text.substr(position, next_position - position));
            output.insert(output.end(), prefix.begin(), prefix.end());
        }
        output.push_back(next_special->second);
        position = next_position + next_special->first.size();
    }
    return output;
}

std::vector<TokenId> Qwen35Tokenizer::encode_normal(std::string_view text) const {
    std::vector<TokenId> output;
    for (const std::string& piece : pretokenize(text)) {
        for (const std::string& bpe_token : apply_bpe(byte_encode(piece))) {
            const auto iterator = token_to_id_.find(bpe_token);
            if (iterator == token_to_id_.end()) {
                throw std::runtime_error("BPE produced token absent from vocabulary");
            }
            output.push_back(iterator->second);
        }
    }
    return output;
}

std::vector<std::string> Qwen35Tokenizer::pretokenize(std::string_view text) const {
    const std::vector<Utf8Unit> units = decode_utf8_units(text);
    std::vector<std::string> pieces;
    std::size_t index = 0;
    const auto append = [&pieces, text, &units](std::size_t first, std::size_t last) {
        pieces.emplace_back(text.substr(units[first].begin,
                                        units[last - 1].end - units[first].begin));
    };

    while (index < units.size()) {
        const std::size_t contraction = contraction_length(text, units[index].begin);
        if (contraction != 0) {
            const std::size_t end_byte = units[index].begin + contraction;
            std::size_t end_index = index;
            while (end_index < units.size() && units[end_index].end <= end_byte) {
                ++end_index;
            }
            if (end_index > index && units[end_index - 1].end == end_byte) {
                append(index, end_index);
                index = end_index;
                continue;
            }
        }

        if (is_letter_or_mark(units[index])) {
            std::size_t end = index + 1;
            while (end < units.size() && is_letter_or_mark(units[end])) {
                ++end;
            }
            append(index, end);
            index = end;
            continue;
        }
        if (!is_newline(units[index]) && !is_letter(units[index]) &&
            !is_number(units[index]) && index + 1 < units.size() &&
            is_letter_or_mark(units[index + 1])) {
            std::size_t end = index + 2;
            while (end < units.size() && is_letter_or_mark(units[end])) {
                ++end;
            }
            append(index, end);
            index = end;
            continue;
        }

        if (is_number(units[index])) {
            append(index, index + 1);
            ++index;
            continue;
        }

        std::size_t punctuation_begin = index;
        std::size_t punctuation = index;
        if (is_ascii_space(units[index]) && index + 1 < units.size() &&
            is_punctuation_class(units[index + 1])) {
            punctuation = index + 1;
        }
        if (punctuation < units.size() && is_punctuation_class(units[punctuation])) {
            std::size_t end = punctuation + 1;
            while (end < units.size() && is_punctuation_class(units[end])) {
                ++end;
            }
            while (end < units.size() && is_newline(units[end])) {
                ++end;
            }
            append(punctuation_begin, end);
            index = end;
            continue;
        }

        if (is_whitespace(units[index])) {
            std::size_t run_end = index;
            std::size_t last_newline = units.size();
            while (run_end < units.size() && is_whitespace(units[run_end])) {
                if (is_newline(units[run_end])) {
                    last_newline = run_end;
                }
                ++run_end;
            }
            if (last_newline != units.size()) {
                append(index, last_newline + 1);
                index = last_newline + 1;
            } else if (run_end < units.size() && run_end - index > 1) {
                // The Qwen35 regex branch \s+(?!\S) keeps the final
                // whitespace code point available as the optional prefix of
                // the following word or punctuation piece.
                append(index, run_end - 1);
                index = run_end - 1;
            } else {
                append(index, run_end);
                index = run_end;
            }
            continue;
        }

        append(index, index + 1);
        ++index;
    }
    return pieces;
}

std::vector<std::string> Qwen35Tokenizer::apply_bpe(std::string encoded_piece) const {
    const std::vector<Utf8Unit> units = decode_utf8_units(encoded_piece);
    std::vector<std::string> symbols;
    symbols.reserve(units.size());
    for (const Utf8Unit& unit : units) {
        symbols.emplace_back(encoded_piece.substr(unit.begin, unit.end - unit.begin));
    }
    if (symbols.size() < 2) {
        return symbols;
    }

    while (symbols.size() > 1) {
        std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
        std::optional<std::pair<std::string, std::string>> best_pair;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const std::pair<std::string, std::string> pair{symbols[index], symbols[index + 1]};
            const auto iterator = merge_ranks_.find(pair);
            if (iterator != merge_ranks_.end() && iterator->second < best_rank) {
                best_rank = iterator->second;
                best_pair = pair;
            }
        }
        if (!best_pair) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() && symbols[index] == best_pair->first &&
                symbols[index + 1] == best_pair->second) {
                merged.push_back(symbols[index] + symbols[index + 1]);
                index += 2;
            } else {
                merged.push_back(std::move(symbols[index]));
                ++index;
            }
        }
        symbols = std::move(merged);
    }
    return symbols;
}

std::string Qwen35Tokenizer::byte_encode(std::string_view bytes) const {
    std::string output;
    output.reserve(bytes.size() * 2);
    for (const char raw_byte : bytes) {
        const auto byte = static_cast<unsigned char>(raw_byte);
        output += byte_to_unicode_[byte];
    }
    return output;
}

std::string Qwen35Tokenizer::byte_decode(std::string_view encoded) const {
    std::string output;
    const std::vector<Utf8Unit> units = decode_utf8_units(encoded);
    output.reserve(units.size());
    for (const Utf8Unit& unit : units) {
        const auto iterator = unicode_to_byte_.find(unit.codepoint);
        if (unit.valid && iterator != unicode_to_byte_.end()) {
            output.push_back(static_cast<char>(iterator->second));
        } else {
            output.append(encoded.substr(unit.begin, unit.end - unit.begin));
        }
    }
    return output;
}

std::string Qwen35Tokenizer::decode(std::span<const TokenId> token_ids,
                                    DecodeOptions options) const {
    std::string output;
    std::string encoded_normal;
    const auto flush_normal = [this, &output, &encoded_normal] {
        if (!encoded_normal.empty()) {
            output += byte_decode(encoded_normal);
            encoded_normal.clear();
        }
    };
    for (TokenId token_id : token_ids) {
        if (token_id >= tokens_.size()) {
            throw std::out_of_range("token id is outside vocabulary");
        }
        if (is_special(token_id)) {
            flush_normal();
            if (!options.skip_special_tokens) {
                output += tokens_[token_id];
            }
        } else if (token_types_[token_id] != TokenType::unused) {
            encoded_normal += tokens_[token_id];
        }
    }
    flush_normal();
    return output;
}

std::size_t Qwen35Tokenizer::vocabulary_size() const noexcept { return tokens_.size(); }
std::size_t Qwen35Tokenizer::merge_count() const noexcept { return merge_ranks_.size(); }

std::string_view Qwen35Tokenizer::token_text(TokenId token_id) const {
    if (token_id >= tokens_.size()) {
        throw std::out_of_range("token id is outside vocabulary");
    }
    return tokens_[token_id];
}

TokenType Qwen35Tokenizer::token_type(TokenId token_id) const {
    if (token_id >= token_types_.size()) {
        throw std::out_of_range("token id is outside vocabulary");
    }
    return token_types_[token_id];
}

std::optional<TokenId> Qwen35Tokenizer::find_token(std::string_view token) const noexcept {
    const auto iterator = token_to_id_.find(std::string(token));
    if (iterator == token_to_id_.end()) {
        return std::nullopt;
    }
    return iterator->second;
}

bool Qwen35Tokenizer::is_special(TokenId token_id) const {
    const TokenType type = token_type(token_id);
    return type == TokenType::control || type == TokenType::user_defined;
}

std::optional<TokenId> Qwen35Tokenizer::bos_token_id() const noexcept {
    return bos_token_id_;
}
std::optional<TokenId> Qwen35Tokenizer::eos_token_id() const noexcept {
    return eos_token_id_;
}
std::optional<TokenId> Qwen35Tokenizer::padding_token_id() const noexcept {
    return padding_token_id_;
}
std::string_view Qwen35Tokenizer::chat_template() const noexcept { return chat_template_; }

std::string token_ids_json(std::span<const TokenId> token_ids) {
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < token_ids.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << token_ids[index];
    }
    output << ']';
    return output.str();
}

}  // namespace oracle::tokenizer
