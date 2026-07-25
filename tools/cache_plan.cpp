#include "oracle/model/gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/runtime/hybrid_cache.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sstream>

namespace {

struct Options {
    std::string path;
    std::size_t tokens{4096};
    bool json{false};
};

[[nodiscard]] std::size_t parse_size(std::string_view text) {
    std::uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0 ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid token count: " + std::string(text));
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument(
            "usage: oracle-cache-plan <model.gguf> [--tokens 4096] [--json]");
    }
    Options options;
    options.path = argv[1];
    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--tokens") {
            if (index + 1 >= argc) {
                throw std::invalid_argument("--tokens requires a value");
            }
            options.tokens = parse_size(argv[++index]);
        } else if (argument == "--json") {
            options.json = true;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}


[[nodiscard]] std::string json_escape(std::string_view value) {
    std::ostringstream output;
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (character < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned int>(character) << std::dec;
                } else {
                    output << static_cast<char>(character);
                }
        }
    }
    return output.str();
}

[[nodiscard]] double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const oracle::model::GgufFile file = oracle::model::GgufReader::read(options.path);
        const oracle::model::Qwen35Manifest manifest =
            oracle::model::load_qwen35_manifest(file);
        const oracle::runtime::Qwen35SsmLayout ssm =
            oracle::runtime::qwen35_ssm_layout(manifest);
        const oracle::runtime::HybridCachePlan plan =
            oracle::runtime::plan_qwen35_cache(manifest, options.tokens);

        if (options.json) {
            std::cout << "{\"model\":\"" << json_escape(manifest.model_name)
                      << "\",\"tokens\":" << plan.maximum_tokens
                      << ",\"attention_layers\":" << plan.attention_layers
                      << ",\"ssm_layers\":" << plan.ssm_layers
                      << ",\"kv_bytes_per_attention_layer\":"
                      << plan.kv_bytes_per_attention_layer
                      << ",\"convolution_bytes_per_ssm_layer\":"
                      << plan.convolution_bytes_per_ssm_layer
                      << ",\"recurrent_bytes_per_ssm_layer\":"
                      << plan.recurrent_bytes_per_ssm_layer
                      << ",\"total_bytes\":" << plan.total_bytes
                      << ",\"ssm_key_heads\":" << ssm.key_heads
                      << ",\"ssm_value_heads\":" << ssm.value_heads
                      << ",\"ssm_key_dimension\":" << ssm.key_head_dimension
                      << ",\"ssm_value_dimension\":" << ssm.value_head_dimension
                      << ",\"ssm_convolution_channels\":" << ssm.convolution_channels
                      << "}\n";
        } else {
            std::cout << "Oracle Qwen3.5 cache plan\n"
                      << "model=" << manifest.model_name
                      << " tokens=" << plan.maximum_tokens << '\n'
                      << "blocks=" << manifest.backbone_block_count
                      << " attention=" << plan.attention_layers
                      << " ssm=" << plan.ssm_layers << '\n'
                      << "kv_per_attention_layer="
                      << plan.kv_bytes_per_attention_layer << " bytes ("
                      << std::fixed << std::setprecision(2)
                      << mib(plan.kv_bytes_per_attention_layer) << " MiB)\n"
                      << "ssm_conv_per_layer="
                      << plan.convolution_bytes_per_ssm_layer << " bytes ("
                      << mib(plan.convolution_bytes_per_ssm_layer) << " MiB)\n"
                      << "ssm_recurrent_per_layer="
                      << plan.recurrent_bytes_per_ssm_layer << " bytes ("
                      << mib(plan.recurrent_bytes_per_ssm_layer) << " MiB)\n"
                      << "total=" << plan.total_bytes << " bytes ("
                      << mib(plan.total_bytes) << " MiB)\n"
                      << "ssm_layout=key_heads:" << ssm.key_heads
                      << " value_heads:" << ssm.value_heads
                      << " key_dim:" << ssm.key_head_dimension
                      << " value_dim:" << ssm.value_head_dimension
                      << " conv_channels:" << ssm.convolution_channels << '\n';
        }
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "cache planning failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
