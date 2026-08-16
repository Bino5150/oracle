#include "oracle/model/mapped_gguf.hpp"
#include "oracle/model/qwen35_manifest.hpp"
#include "oracle/model/qwen35_weights.hpp"
#include "oracle/runtime/qwen35_chat.hpp"
#include "oracle/runtime/qwen35_generation.hpp"
#include "oracle/tokenizer/qwen35_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Phase 2E Slice 2 fixed, versioned known-answer fixture. Changing any of
// these values changes the fixture identity and must bump the id below.
constexpr std::string_view kFixtureId = "oracle-phase2e-slice2-fixture-v1";
constexpr std::string_view kDefaultChatSystem = "You are a helpful assistant.";
constexpr std::string_view kDefaultChatUser = "What is the capital of France?";
constexpr std::size_t kDefaultMaxGeneratedTokens = 16;
constexpr std::size_t kDefaultMaximumContextTokens = 256;
constexpr std::size_t kDefaultRepeat = 2;
constexpr std::string_view kOracleBaselineCommit =
    "5ecf054f1409fc5cdcf4198cb91a5acfb3ea4d90";
constexpr std::string_view kOracleBaselineVersion = "1.0.0-phase2d";
constexpr std::string_view kOracleCandidate = "phase2e-slice3c";

struct Options {
    std::filesystem::path model_path;
    std::string chat_system{kDefaultChatSystem};
    std::string chat_user{kDefaultChatUser};
    std::size_t max_generated_tokens{kDefaultMaxGeneratedTokens};
    std::size_t maximum_context_tokens{kDefaultMaximumContextTokens};
    std::size_t repeat{kDefaultRepeat};
    std::optional<std::string> model_sha256;
    std::optional<std::filesystem::path> fixture_json_path;
    std::optional<std::filesystem::path> events_json_path;
    bool stream_events{false};
    bool json{false};
    std::vector<std::string> text_stops;
    std::vector<std::vector<std::uint32_t>> token_stops;

    // Phase 2E Slice 3C: reasoning-loop safety validation controls. A
    // reasoning boundary is only configured (and therefore only capable of
    // activating regardless of policy) when both start/end token lists are
    // provided.
    oracle::runtime::Qwen35ReasoningLoopPolicy reasoning_policy{
        oracle::runtime::Qwen35ReasoningLoopPolicy::off};
    std::optional<std::vector<std::uint32_t>> reasoning_start_tokens;
    std::optional<std::vector<std::uint32_t>> reasoning_end_tokens;
    std::optional<std::size_t> reasoning_min_tokens;
    std::optional<std::size_t> reasoning_window;
    std::optional<std::size_t> reasoning_max_period;
    std::optional<std::size_t> reasoning_min_coverage;
    std::optional<std::size_t> reasoning_max_interventions;
    // Validation-only escape hatch (see Qwen35ReasoningLoopConfig in
    // qwen35_generation.hpp) -- exists so the real-model force-close gate
    // can be exercised deterministically and cheaply without waiting for
    // (or trying to manufacture) genuine pathological repetition. Never a
    // production flag.
    std::optional<std::size_t> reasoning_validation_trigger_at;
};

[[noreturn]] void usage() {
    throw std::invalid_argument(
        "usage: oracle-qwen35-generate <model.gguf> "
        "[--chat-system <text>] [--chat-user <text>] "
        "[--max-generated-tokens <n>] [--maximum-context-tokens <n>] "
        "[--repeat <n>] [--model-sha256 <hex>] [--fixture-json <path>] "
        "[--stream-events] [--events-json <path>] "
        "[--stop <text>]... [--stop-token-ids <id,id,...>]... [--json] "
        "[--reasoning-loop-policy off|stop|force-close] "
        "[--reasoning-start-tokens <id,id,...>] [--reasoning-end-tokens <id,id,...>] "
        "[--reasoning-min-tokens <n>] [--reasoning-window <n>] "
        "[--reasoning-max-period <n>] [--reasoning-min-coverage <n>] "
        "[--reasoning-max-interventions <n>] "
        "[--reasoning-validation-trigger-at <n>]");
}

[[nodiscard]] std::vector<std::uint32_t> parse_token_id_list(std::string_view text) {
    std::vector<std::uint32_t> ids;
    std::size_t position = 0;
    while (position < text.size()) {
        while (position < text.size() && (text[position] == ',' || text[position] == ' ')) {
            ++position;
        }
        if (position == text.size()) break;
        const std::size_t begin = position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') {
            ++position;
        }
        if (position == begin) {
            throw std::invalid_argument(
                "--stop-token-ids requires a comma-separated list of non-negative integers");
        }
        const std::string token_text(text.substr(begin, position - begin));
        ids.push_back(static_cast<std::uint32_t>(std::stoul(token_text)));
        if (position < text.size() && text[position] != ',') {
            throw std::invalid_argument("--stop-token-ids entries must be comma-separated");
        }
    }
    if (ids.empty()) {
        throw std::invalid_argument("--stop-token-ids must not be empty");
    }
    return ids;
}

[[nodiscard]] std::size_t parse_size(std::string_view text, std::string_view name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(std::string(text), &consumed, 10);
    if (consumed != text.size()) {
        throw std::invalid_argument(std::string(name) + " contains trailing characters");
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
    if (argc < 2) usage();
    Options options;
    options.model_path = argv[1];

    for (int index = 2; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto require_value = [argc, argv, &index, argument]() -> std::string {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--chat-system") {
            options.chat_system = require_value();
        } else if (argument == "--chat-user") {
            options.chat_user = require_value();
        } else if (argument == "--max-generated-tokens") {
            options.max_generated_tokens = parse_size(require_value(), "max-generated-tokens");
        } else if (argument == "--maximum-context-tokens") {
            options.maximum_context_tokens =
                parse_size(require_value(), "maximum-context-tokens");
        } else if (argument == "--repeat") {
            options.repeat = parse_size(require_value(), "repeat");
        } else if (argument == "--model-sha256") {
            options.model_sha256 = require_value();
        } else if (argument == "--fixture-json") {
            options.fixture_json_path = std::filesystem::path(require_value());
        } else if (argument == "--stream-events") {
            options.stream_events = true;
        } else if (argument == "--events-json") {
            options.events_json_path = std::filesystem::path(require_value());
        } else if (argument == "--stop") {
            options.text_stops.push_back(require_value());
        } else if (argument == "--stop-token-ids") {
            options.token_stops.push_back(parse_token_id_list(require_value()));
        } else if (argument == "--json") {
            options.json = true;
        } else if (argument == "--reasoning-loop-policy") {
            const std::string value = require_value();
            if (value == "off") {
                options.reasoning_policy = oracle::runtime::Qwen35ReasoningLoopPolicy::off;
            } else if (value == "stop") {
                options.reasoning_policy = oracle::runtime::Qwen35ReasoningLoopPolicy::stop;
            } else if (value == "force-close") {
                options.reasoning_policy = oracle::runtime::Qwen35ReasoningLoopPolicy::force_close;
            } else {
                throw std::invalid_argument(
                    "--reasoning-loop-policy must be one of off|stop|force-close");
            }
        } else if (argument == "--reasoning-start-tokens") {
            options.reasoning_start_tokens = parse_token_id_list(require_value());
        } else if (argument == "--reasoning-end-tokens") {
            options.reasoning_end_tokens = parse_token_id_list(require_value());
        } else if (argument == "--reasoning-min-tokens") {
            options.reasoning_min_tokens = parse_size(require_value(), "reasoning-min-tokens");
        } else if (argument == "--reasoning-window") {
            options.reasoning_window = parse_size(require_value(), "reasoning-window");
        } else if (argument == "--reasoning-max-period") {
            options.reasoning_max_period = parse_size(require_value(), "reasoning-max-period");
        } else if (argument == "--reasoning-min-coverage") {
            options.reasoning_min_coverage = parse_size(require_value(), "reasoning-min-coverage");
        } else if (argument == "--reasoning-max-interventions") {
            options.reasoning_max_interventions =
                parse_size(require_value(), "reasoning-max-interventions");
        } else if (argument == "--reasoning-validation-trigger-at") {
            options.reasoning_validation_trigger_at =
                parse_size(require_value(), "reasoning-validation-trigger-at");
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.max_generated_tokens == 0) {
        throw std::invalid_argument("--max-generated-tokens must be at least 1");
    }
    if (options.repeat == 0) {
        throw std::invalid_argument("--repeat must be at least 1");
    }
    return options;
}

[[nodiscard]] std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 16);
    constexpr char digits[] = "0123456789abcdef";
    for (const char raw_character : value) {
        const auto character = static_cast<unsigned char>(raw_character);
        switch (character) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (character < 0x20U) {
                    output += "\\u00";
                    output.push_back(digits[(character >> 4U) & 0x0FU]);
                    output.push_back(digits[character & 0x0FU]);
                } else {
                    output.push_back(static_cast<char>(character));
                }
        }
    }
    return output;
}

struct RunResult {
    std::vector<oracle::runtime::Qwen35GeneratedToken> generated_tokens;
    std::vector<oracle::tokenizer::TokenId> generated_token_ids;
    std::string decoded_text;
    oracle::runtime::Qwen35FinishReason finish_reason{};
    std::size_t final_sequence_length{0};
    std::vector<oracle::runtime::Qwen35GenerationEvent> events;
    std::optional<oracle::runtime::Qwen35StopMatch> stop_match;
    bool reasoning_active_at_finish{false};
    std::vector<oracle::runtime::Qwen35ReasoningIntervention> reasoning_interventions;
};

[[nodiscard]] RunResult run_once(const oracle::model::Qwen35Manifest& manifest,
                                  const oracle::model::Qwen35Weights& weights,
                                  const oracle::tokenizer::Qwen35Tokenizer& tokenizer,
                                  const oracle::runtime::Qwen35GenerationRequest& request,
                                  std::size_t maximum_context_tokens,
                                  bool capture_events,
                                  bool print_events_live) {
    oracle::runtime::Qwen35GenerationSession session(manifest, weights, tokenizer,
                                                      maximum_context_tokens);
    RunResult run;

    oracle::runtime::Qwen35GenerationCallback callback;
    if (capture_events || print_events_live) {
        callback = [&run, print_events_live](const oracle::runtime::Qwen35GenerationEvent& event) {
            // Diagnostics/streaming go to stderr in text mode and as
            // dedicated JSON lines on stdout only when explicitly requested,
            // so they never corrupt the tool's primary machine-readable
            // summary output.
            run.events.push_back(event);
            if (print_events_live) {
                std::cout << oracle::runtime::qwen35_generation_event_json(event) << '\n';
            }
        };
    }

    const oracle::runtime::Qwen35GenerationResult result =
        session.generate_fresh(request, callback);

    run.generated_tokens = result.generated_tokens;
    run.generated_token_ids.reserve(result.generated_tokens.size());
    for (const oracle::runtime::Qwen35GeneratedToken& token : result.generated_tokens) {
        run.generated_token_ids.push_back(token.token_id);
    }
    // Single authoritative decoding path: Slice 3A's event decoder produces
    // this field inside generate_fresh(); this tool does not independently
    // re-decode the generated tokens.
    run.decoded_text = result.generated_text;
    run.finish_reason = result.finish_reason;
    run.final_sequence_length = result.final_sequence_length;
    run.stop_match = result.stop_match;
    run.reasoning_active_at_finish = result.reasoning_active_at_finish;
    run.reasoning_interventions = result.reasoning_interventions;

    if (run.final_sequence_length !=
        result.prompt_tokens.size() + result.generated_tokens.size()) {
        throw std::runtime_error(
            "Qwen3.5 generation fixture ledger mismatch: final_sequence_length does not equal "
            "prompt_tokens.size() + generated_tokens.size()");
    }
    if (capture_events && run.events.size() != run.generated_tokens.size()) {
        throw std::runtime_error(
            "Qwen3.5 generation fixture: event count does not match generated token count");
    }
    return run;
}

[[nodiscard]] bool token_ids_equal(const std::vector<oracle::tokenizer::TokenId>& left,
                                    const std::vector<oracle::tokenizer::TokenId>& right) {
    return left == right;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const bool capture_events =
            options.events_json_path.has_value() || options.stream_events;

        const oracle::model::MappedGgufModel model(options.model_path);
        const oracle::model::Qwen35Manifest manifest =
            oracle::model::load_qwen35_manifest(model.file());
        const oracle::model::Qwen35Weights weights =
            oracle::model::bind_qwen35_weights(model, manifest);
        const oracle::tokenizer::Qwen35Tokenizer tokenizer(model.file());

        oracle::runtime::Qwen35ChatRequest chat;
        chat.add_generation_prompt = true;
        if (!options.chat_system.empty()) {
            chat.messages.push_back(
                {oracle::runtime::ChatRole::system, options.chat_system, {}, {}});
        }
        chat.messages.push_back({oracle::runtime::ChatRole::user, options.chat_user, {}, {}});

        const std::string formatted_prompt = oracle::runtime::format_qwen35_chat(chat);

        oracle::runtime::SamplingConfig sampling;
        sampling.temperature = 0.0F;
        oracle::runtime::Qwen35GenerationRequest request =
            oracle::runtime::make_qwen35_chat_generation_request(
                tokenizer, chat, options.max_generated_tokens, sampling);
        request.text_stop_sequences = options.text_stops;
        request.token_stop_sequences = options.token_stops;

        if (options.reasoning_start_tokens.has_value() && options.reasoning_end_tokens.has_value()) {
            oracle::runtime::Qwen35ReasoningBoundary boundary;
            boundary.start_tokens = *options.reasoning_start_tokens;
            boundary.end_tokens = *options.reasoning_end_tokens;
            boundary.force_close_supported = true;
            request.reasoning_boundary = boundary;
        }
        request.reasoning_loop.policy = options.reasoning_policy;
        if (options.reasoning_min_tokens.has_value()) {
            request.reasoning_loop.minimum_reasoning_tokens = *options.reasoning_min_tokens;
        }
        if (options.reasoning_window.has_value()) {
            request.reasoning_loop.inspection_window = *options.reasoning_window;
        }
        if (options.reasoning_max_period.has_value()) {
            request.reasoning_loop.maximum_period = *options.reasoning_max_period;
        }
        if (options.reasoning_min_coverage.has_value()) {
            request.reasoning_loop.minimum_repeated_coverage = *options.reasoning_min_coverage;
        }
        if (options.reasoning_max_interventions.has_value()) {
            request.reasoning_loop.maximum_interventions = *options.reasoning_max_interventions;
        }
        request.reasoning_loop.validation_only_force_trigger_at_reasoning_token_count =
            options.reasoning_validation_trigger_at;

        if (request.prompt_tokens.empty()) {
            throw std::runtime_error("Qwen3.5 generation fixture prompt tokenized to zero tokens");
        }

        std::vector<RunResult> runs;
        runs.reserve(options.repeat);
        for (std::size_t attempt = 0; attempt < options.repeat; ++attempt) {
            runs.push_back(run_once(manifest, weights, tokenizer, request,
                                    options.maximum_context_tokens, capture_events,
                                    options.stream_events && !options.json));
        }

        bool all_token_ids_identical = true;
        bool all_decoded_text_identical = true;
        for (std::size_t index = 1; index < runs.size(); ++index) {
            if (!token_ids_equal(runs[0].generated_token_ids, runs[index].generated_token_ids)) {
                all_token_ids_identical = false;
            }
            if (runs[0].decoded_text != runs[index].decoded_text) {
                all_decoded_text_identical = false;
            }
        }
        // Text equality is only a meaningful signal once token-ID equality holds;
        // decode() is a pure function of the token IDs, so this should never fire
        // when all_token_ids_identical is true, but we check independently and
        // surface both because token IDs are the authoritative comparison.
        if (all_token_ids_identical && !all_decoded_text_identical) {
            throw std::runtime_error(
                "Qwen3.5 generation fixture: identical token IDs decoded to different text");
        }

        if (options.events_json_path.has_value()) {
            std::ofstream events_output(*options.events_json_path);
            if (!events_output) {
                throw std::runtime_error("unable to open events-json output: " +
                                         options.events_json_path->string());
            }
            events_output << "{\"runs\":[";
            for (std::size_t run_index = 0; run_index < runs.size(); ++run_index) {
                if (run_index != 0) events_output << ',';
                events_output << '[';
                const std::vector<oracle::runtime::Qwen35GenerationEvent>& events =
                    runs[run_index].events;
                for (std::size_t event_index = 0; event_index < events.size(); ++event_index) {
                    if (event_index != 0) events_output << ',';
                    events_output << oracle::runtime::qwen35_generation_event_json(
                        events[event_index]);
                }
                events_output << ']';
            }
            events_output << "]}\n";
            if (!events_output) {
                throw std::runtime_error("failed while writing events-json output");
            }
        }

        std::ostringstream json;
        json << '{'
             << "\"fixture_id\":\"" << json_escape(kFixtureId) << "\","
             << "\"oracle_baseline_commit\":\"" << json_escape(kOracleBaselineCommit) << "\","
             << "\"oracle_baseline_version\":\"" << json_escape(kOracleBaselineVersion) << "\","
             << "\"oracle_candidate\":\"" << json_escape(kOracleCandidate) << "\","
             << "\"model_path\":\"" << json_escape(options.model_path.string()) << "\",";
        json << "\"model_sha256\":";
        if (options.model_sha256.has_value()) {
            json << '"' << json_escape(*options.model_sha256) << '"';
        } else {
            json << "null";
        }
        json << ','
             << "\"numerical_contract\":{"
             << "\"execution_projection\":\"decoded-f32-scalar\","
             << "\"execution_attention_cache\":\"f32-semantic\"},"
             << "\"sampling\":{\"strategy\":\"greedy\",\"temperature\":0.0},"
             << "\"maximum_context_tokens\":" << options.maximum_context_tokens << ','
             << "\"max_generated_tokens\":" << options.max_generated_tokens << ','
             << "\"reasoning_loop_policy\":\""
             << json_escape(oracle::runtime::qwen35_reasoning_loop_policy_name(
                    request.reasoning_loop.policy))
             << "\","
             << "\"chat_request\":{"
             << "\"system\":";
        if (options.chat_system.empty()) {
            json << "null";
        } else {
            json << '"' << json_escape(options.chat_system) << '"';
        }
        json << ','
             << "\"user\":\"" << json_escape(options.chat_user) << "\"},"
             << "\"formatted_prompt_text\":\"" << json_escape(formatted_prompt) << "\","
             << "\"prompt_token_count\":" << request.prompt_tokens.size() << ','
             << "\"prompt_token_ids\":"
             << oracle::tokenizer::token_ids_json(request.prompt_tokens) << ','
             << "\"runs\":[";
        for (std::size_t index = 0; index < runs.size(); ++index) {
            if (index != 0) json << ',';
            const RunResult& run = runs[index];
            json << '{'
                 << "\"finish_reason\":\""
                 << json_escape(oracle::runtime::qwen35_finish_reason_name(run.finish_reason))
                 << "\","
                 << "\"final_sequence_length\":" << run.final_sequence_length << ','
                 << "\"generated_token_count\":" << run.generated_tokens.size() << ','
                 << "\"generated_tokens\":[";
            for (std::size_t token_index = 0; token_index < run.generated_tokens.size();
                 ++token_index) {
                if (token_index != 0) json << ',';
                const oracle::runtime::Qwen35GeneratedToken& token =
                    run.generated_tokens[token_index];
                json << '{'
                     << "\"token_id\":" << token.token_id << ','
                     << "\"position\":" << token.position << ','
                     << "\"probability\":" << token.probability << ','
                     << "\"candidate_count\":" << token.candidate_count << ','
                     << "\"token_text\":\"" << json_escape(tokenizer.token_text(token.token_id))
                     << "\"}";
            }
            json << "],\"decoded_text\":\"" << json_escape(run.decoded_text) << "\","
                 << "\"stop_match\":";
            if (run.stop_match.has_value()) {
                json << oracle::runtime::qwen35_stop_match_json(*run.stop_match);
            } else {
                json << "null";
            }
            json << ',' << "\"reasoning_active_at_finish\":"
                 << (run.reasoning_active_at_finish ? "true" : "false") << ','
                 << "\"reasoning_intervention_count\":" << run.reasoning_interventions.size()
                 << ',' << "\"reasoning_interventions\":[";
            for (std::size_t intervention_index = 0;
                 intervention_index < run.reasoning_interventions.size(); ++intervention_index) {
                if (intervention_index != 0) json << ',';
                json << oracle::runtime::qwen35_reasoning_intervention_json(
                    run.reasoning_interventions[intervention_index]);
            }
            json << ']' << '}';
        }
        json << "],"
             << "\"determinism\":{"
             << "\"repeat_count\":" << runs.size() << ','
             << "\"all_runs_token_id_identical\":"
             << (all_token_ids_identical ? "true" : "false") << ','
             << "\"all_runs_decoded_text_identical\":"
             << (all_decoded_text_identical ? "true" : "false") << "}}";

        const std::string json_text = json.str();
        if (options.fixture_json_path.has_value()) {
            std::ofstream output(*options.fixture_json_path);
            if (!output) {
                throw std::runtime_error("unable to open fixture-json output: " +
                                         options.fixture_json_path->string());
            }
            output << json_text << '\n';
            if (!output) {
                throw std::runtime_error("failed while writing fixture-json output");
            }
        }

        if (options.json) {
            std::cout << json_text << '\n';
        } else {
            std::cout << "Oracle Qwen3.5 real generation fixture: " << kFixtureId << '\n'
                       << "baseline: " << kOracleBaselineCommit << " (" << kOracleBaselineVersion
                       << "), candidate: " << kOracleCandidate << '\n'
                       << "model: " << options.model_path.string() << '\n'
                       << "prompt tokens: " << request.prompt_tokens.size()
                       << ", max generated tokens: " << options.max_generated_tokens
                       << ", maximum context tokens: " << options.maximum_context_tokens << '\n'
                       << "formatted prompt:\n" << formatted_prompt << "\n---\n";
            for (std::size_t index = 0; index < runs.size(); ++index) {
                const RunResult& run = runs[index];
                std::cout << "run " << index << ": finish_reason="
                           << oracle::runtime::qwen35_finish_reason_name(run.finish_reason)
                           << " generated_tokens=" << run.generated_tokens.size()
                           << " final_sequence_length=" << run.final_sequence_length << '\n'
                           << "  token_ids=";
                for (std::size_t token_index = 0; token_index < run.generated_token_ids.size();
                     ++token_index) {
                    if (token_index != 0) std::cout << ',';
                    std::cout << run.generated_token_ids[token_index];
                }
                std::cout << '\n' << "  decoded_text=" << run.decoded_text << '\n';
                if (run.stop_match.has_value()) {
                    std::cout << "  stop_match: kind="
                               << oracle::runtime::qwen35_stop_kind_name(run.stop_match->kind)
                               << " configured_index=" << run.stop_match->configured_index
                               << " generated_token_range=["
                               << run.stop_match->generated_token_begin << ','
                               << run.stop_match->generated_token_end << ')'
                               << " text_byte_offset=" << run.stop_match->text_byte_offset
                               << " matched_text=" << run.stop_match->matched_text << '\n';
                }
                if (!run.reasoning_interventions.empty()) {
                    std::cout << "  reasoning_active_at_finish="
                               << (run.reasoning_active_at_finish ? "true" : "false")
                               << " reasoning_intervention_count="
                               << run.reasoning_interventions.size() << '\n';
                    for (const auto& intervention : run.reasoning_interventions) {
                        std::cout << "    intervention #" << intervention.intervention_number
                                   << ": generated_index=" << intervention.generated_index
                                   << " reasoning_token_count=" << intervention.reasoning_token_count
                                   << " detected_period=" << intervention.detected_period
                                   << " repeated_coverage=" << intervention.repeated_coverage
                                   << " policy="
                                   << oracle::runtime::qwen35_reasoning_loop_policy_name(
                                          intervention.policy)
                                   << " closure_attempted="
                                   << (intervention.closure_attempted ? "true" : "false")
                                   << " closure_succeeded="
                                   << (intervention.closure_succeeded ? "true" : "false")
                                   << " failure_reason=" << intervention.failure_reason << '\n';
                    }
                }
            }
            std::cout << "determinism: repeat_count=" << runs.size()
                       << " all_runs_token_id_identical="
                       << (all_token_ids_identical ? "true" : "false")
                       << " all_runs_decoded_text_identical="
                       << (all_decoded_text_identical ? "true" : "false") << '\n';
        }

        return all_token_ids_identical ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "oracle-qwen35-generate: " << error.what() << '\n';
        return 1;
    }
}
