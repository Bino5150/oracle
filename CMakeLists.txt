cmake_minimum_required(VERSION 3.24)
project(oracle_inference_engine VERSION 1.0.0 LANGUAGES CXX)

option(ORACLE_ENABLE_CUDA "Build the CUDA backend" OFF)
option(ORACLE_BUILD_TESTS "Build tests" ON)
option(ORACLE_BUILD_EXAMPLES "Build examples" ON)
option(ORACLE_BUILD_BENCHMARKS "Build the benchmark harness" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(oracle_engine
    src/core/config.cpp
    src/core/mapped_file.cpp
    src/core/memory_arena.cpp
    src/core/scratch_planner.cpp
    src/core/tensor.cpp
    src/backend/cpu/cpu_backend.cpp
    src/backend/cpu/quantized_reference.cpp
    src/model/ggml_type.cpp
    src/model/gguf.cpp
    src/model/mapped_gguf.cpp
    src/model/qwen35_manifest.cpp
    src/model/qwen35_weights.cpp
    src/model/storage_decode.cpp
    src/model/model.cpp
    src/runtime/engine.cpp
    src/runtime/reference_pipeline.cpp
    src/runtime/qwen35_chat.cpp
    src/runtime/qwen35_block.cpp
    src/runtime/qwen35_generation.cpp
    src/runtime/qwen35_forward.cpp
    src/runtime/hybrid_cache.cpp
    src/runtime/reference_kernels.cpp
    src/runtime/sampler.cpp
    src/runtime/tiny_hybrid_model.cpp
    src/scheduler/scheduler.cpp
    src/tokenizer/qwen35_tokenizer.cpp
)

target_include_directories(oracle_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(oracle_engine PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(oracle_engine PRIVATE /W4 /permissive-)
else()
    target_compile_options(oracle_engine PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

if(ORACLE_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(oracle_engine PRIVATE src/backend/cuda/cuda_backend.cu)
    target_compile_definitions(oracle_engine PUBLIC ORACLE_ENABLE_CUDA=1)
    set_target_properties(oracle_engine PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
endif()

add_executable(oracle-cli src/main.cpp)
target_link_libraries(oracle-cli PRIVATE oracle_engine)

add_executable(oracle-gguf-inspect tools/gguf_inspect.cpp)
target_link_libraries(oracle-gguf-inspect PRIVATE oracle_engine)

add_executable(oracle-tokenize tools/tokenize.cpp)
target_link_libraries(oracle-tokenize PRIVATE oracle_engine)

add_executable(oracle-reference-decode tools/reference_decode.cpp)
target_link_libraries(oracle-reference-decode PRIVATE oracle_engine)

add_executable(oracle-cache-plan tools/cache_plan.cpp)
target_link_libraries(oracle-cache-plan PRIVATE oracle_engine)

add_executable(oracle-storage-row tools/storage_row.cpp)
target_link_libraries(oracle-storage-row PRIVATE oracle_engine)

add_executable(oracle-qwen35-bind tools/qwen35_bind.cpp)
target_link_libraries(oracle-qwen35-bind PRIVATE oracle_engine)

add_executable(oracle-qwen35-block tools/qwen35_block.cpp)
target_link_libraries(oracle-qwen35-block PRIVATE oracle_engine)

add_executable(oracle-qwen35-forward tools/qwen35_forward.cpp)
target_link_libraries(oracle-qwen35-forward PRIVATE oracle_engine)

add_executable(oracle-qwen35-generate tools/qwen35_generate.cpp)
target_link_libraries(oracle-qwen35-generate PRIVATE oracle_engine)

if(ORACLE_BUILD_EXAMPLES)
    add_executable(oracle-smoke examples/smoke.cpp)
    target_link_libraries(oracle-smoke PRIVATE oracle_engine)

    add_executable(oracle-reference-pipeline examples/reference_pipeline.cpp)
    target_link_libraries(oracle-reference-pipeline PRIVATE oracle_engine)
endif()

if(ORACLE_BUILD_BENCHMARKS)
    add_executable(oracle-bench benchmarks/oracle_bench.cpp)
    target_link_libraries(oracle-bench PRIVATE oracle_engine)
endif()

if(ORACLE_BUILD_TESTS)
    enable_testing()
    add_executable(oracle-tests tests/test_engine.cpp)
    target_link_libraries(oracle-tests PRIVATE oracle_engine)
    add_test(NAME oracle-tests COMMAND oracle-tests)

    add_executable(oracle-phase1b-tests tests/test_phase1b.cpp)
    target_link_libraries(oracle-phase1b-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase1b-tests COMMAND oracle-phase1b-tests)

    add_executable(oracle-phase1c-tests tests/test_phase1c.cpp)
    target_link_libraries(oracle-phase1c-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase1c-tests COMMAND oracle-phase1c-tests)

    add_executable(oracle-phase1c1-tests tests/test_phase1c1.cpp)
    target_link_libraries(oracle-phase1c1-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase1c1-tests COMMAND oracle-phase1c1-tests)

    add_executable(oracle-phase1d-tests tests/test_phase1d.cpp)
    target_link_libraries(oracle-phase1d-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase1d-tests COMMAND oracle-phase1d-tests)

    add_executable(oracle-phase1e-tests tests/test_phase1e.cpp)
    target_link_libraries(oracle-phase1e-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase1e-tests COMMAND oracle-phase1e-tests)

    add_executable(oracle-phase2a-tests tests/test_phase2a.cpp)
    target_link_libraries(oracle-phase2a-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase2a-tests COMMAND oracle-phase2a-tests)

    add_executable(oracle-phase2b-tests tests/test_phase2b.cpp)
    target_link_libraries(oracle-phase2b-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase2b-tests COMMAND oracle-phase2b-tests)

    add_executable(oracle-phase2c-tests tests/test_phase2c.cpp)
    target_link_libraries(oracle-phase2c-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase2c-tests COMMAND oracle-phase2c-tests)

    add_executable(oracle-phase2d-tests tests/test_phase2d.cpp)
    target_link_libraries(oracle-phase2d-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase2d-tests COMMAND oracle-phase2d-tests)

    add_executable(oracle-phase2e-tests tests/test_phase2e.cpp)
    target_link_libraries(oracle-phase2e-tests PRIVATE oracle_engine)
    add_test(NAME oracle-phase2e-tests COMMAND oracle-phase2e-tests)
endif()
