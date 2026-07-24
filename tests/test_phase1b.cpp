#include "oracle/core/memory_arena.hpp"
#include "oracle/core/scratch_planner.hpp"
#include "oracle/model/gguf.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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

std::filesystem::path write_fixture() {
    const auto path = std::filesystem::temp_directory_path() / "oracle-phase1b-fixture.gguf";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("failed to create GGUF test fixture");
    }

    output.write("GGUF", 4);
    write_u32(output, 3);  // version
    write_u64(output, 1);  // tensors
    write_u64(output, 5);  // metadata

    write_string(output, "general.architecture");
    write_u32(output, 8);  // string
    write_string(output, "oracle-test");

    write_string(output, "general.alignment");
    write_u32(output, 4);  // uint32
    write_u32(output, 64);

    write_string(output, "oracle.context_length");
    write_u32(output, 4);  // uint32
    write_u32(output, 4096);

    write_string(output, "oracle.enabled");
    write_u32(output, 7);  // bool
    write_u8(output, 1);

    write_string(output, "tokenizer.ggml.tokens");
    write_u32(output, 9);  // array
    write_u32(output, 8);  // string element type
    write_u64(output, 2);
    write_string(output, "hello");
    write_string(output, "world");

    write_string(output, "token_embd.weight");
    write_u32(output, 2);  // dimensions
    write_u64(output, 4);
    write_u64(output, 3);
    write_u32(output, 0);  // GGML F32
    write_u64(output, 0);  // relative tensor offset

    const std::streamoff current = output.tellp();
    const std::uint64_t aligned =
        (static_cast<std::uint64_t>(current) + 63ULL) & ~63ULL;
    for (std::uint64_t index = static_cast<std::uint64_t>(current); index < aligned; ++index) {
        output.put('\0');
    }
    for (std::size_t index = 0; index < 12; ++index) {
        const float value = static_cast<float>(index);
        write_u32(output, std::bit_cast<std::uint32_t>(value));
    }
    output.close();
    return path;
}

void test_arena(TestRunner& runner) {
    oracle::core::MemoryArena arena(512, 64);
    auto* first = static_cast<std::byte*>(arena.allocate_bytes(17, 16));
    auto* second = arena.allocate<std::uint64_t>(4);
    runner.expect(reinterpret_cast<std::uintptr_t>(first) % 16 == 0,
                  "arena honors requested alignment");
    runner.expect(reinterpret_cast<std::uintptr_t>(second) % alignof(std::uint64_t) == 0,
                  "typed arena allocation is aligned");
    runner.expect(arena.stats().allocation_count == 2, "arena counts allocations");
    runner.expect(arena.stats().peak_bytes == arena.used(), "arena records peak usage");

    const std::size_t before_scope = arena.used();
    {
        oracle::core::ScopedArenaMark scope(arena);
        static_cast<void>(arena.allocate_bytes(96, 32));
        runner.expect(arena.used() > before_scope, "scoped scratch advances arena");
    }
    runner.expect(arena.used() == before_scope, "scoped scratch rewinds arena");

    runner.expect_throws([&arena] { static_cast<void>(arena.allocate_bytes(1024)); },
                         "arena rejects allocations beyond capacity");
    runner.expect_throws([&arena] { arena.rewind(arena.used() + 1); },
                         "arena rejects forward rewind marks");
}

void test_scratch_plan(TestRunner& runner) {
    oracle::core::ScratchPlanner planner;
    planner.add({"a", 64, 64, 0, 1});
    planner.add({"b", 32, 32, 0, 0});
    planner.add({"c", 32, 32, 1, 2});
    planner.add({"d", 16, 16, 3, 3});

    const oracle::core::ScratchPlan plan = planner.build();
    const auto* a = plan.find("a");
    const auto* b = plan.find("b");
    const auto* c = plan.find("c");
    const auto* d = plan.find("d");
    runner.expect(a != nullptr && b != nullptr && c != nullptr && d != nullptr,
                  "scratch plan contains every request");
    runner.expect(a->offset == 0, "first scratch request starts at zero");
    runner.expect(b->offset == 64, "overlapping scratch request follows first");
    runner.expect(c->offset == 64, "non-overlapping lifetime reuses scratch range");
    runner.expect(d->offset == 0, "fully expired scratch range is reused from zero");
    runner.expect(plan.peak_bytes() == 96, "scratch plan reports reusable peak");

    runner.expect_throws([&planner] { planner.add({"a", 8, 8, 0, 0}); },
                         "scratch request names are unique");
}

void test_gguf(TestRunner& runner) {
    const std::filesystem::path path = write_fixture();
    const oracle::model::GgufFile file = oracle::model::GgufReader::read(path);
    runner.expect(file.version == 3, "GGUF version parsed");
    runner.expect(file.tensor_count == 1, "GGUF tensor count parsed");
    runner.expect(file.metadata_count == 5, "GGUF metadata count parsed");
    runner.expect(file.alignment == 64, "GGUF alignment metadata applied");
    runner.expect(file.data_offset % 64 == 0, "GGUF data offset aligned");

    const auto* architecture = file.find_metadata("general.architecture");
    runner.expect(architecture != nullptr, "GGUF metadata lookup");
    runner.expect(architecture != nullptr &&
                      *architecture->value.get_if<std::string>() == "oracle-test",
                  "GGUF string metadata parsed");

    const auto* enabled = file.find_metadata("oracle.enabled");
    runner.expect(enabled != nullptr && *enabled->value.get_if<bool>(),
                  "GGUF boolean metadata parsed");

    const auto* tokens = file.find_metadata("tokenizer.ggml.tokens");
    const auto* token_array = tokens == nullptr
                                  ? nullptr
                                  : tokens->value.get_if<std::shared_ptr<oracle::model::GgufArray>>();
    runner.expect(token_array != nullptr && (*token_array)->values.size() == 2,
                  "GGUF array metadata parsed");
    runner.expect(token_array != nullptr &&
                      *(*token_array)->values[1].get_if<std::string>() == "world",
                  "GGUF array values parsed");

    const auto* tensor = file.find_tensor("token_embd.weight");
    runner.expect(tensor != nullptr, "GGUF tensor lookup");
    runner.expect(tensor != nullptr &&
                      tensor->dimensions == std::vector<std::uint64_t>({4, 3}),
                  "GGUF tensor dimensions parsed");
    runner.expect(tensor != nullptr && tensor->ggml_type == 0,
                  "GGUF tensor type parsed");
    runner.expect(oracle::model::gguf_summary_json(file).find("oracle-test") !=
                      std::string::npos,
                  "GGUF JSON summary includes metadata");

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    TestRunner runner;
    try {
        test_arena(runner);
        test_scratch_plan(runner);
        test_gguf(runner);
    } catch (const std::exception& error) {
        std::cerr << "UNCAUGHT TEST ERROR: " << error.what() << '\n';
        return 1;
    }

    if (runner.failures() == 0) {
        std::cout << "all Phase 1B tests passed\n";
    }
    return runner.result();
}
