cmake_minimum_required(VERSION 3.24)
project(oracle_inference_engine VERSION 0.4.0 LANGUAGES CXX)

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
    src/model/ggml_type.cpp
    src/model/gguf.cpp
    src/model/mapped_gguf.cpp
    src/model/model.cpp
    src/runtime/engine.cpp
    src/runtime/reference_pipeline.cpp
    src/scheduler/scheduler.cpp
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
endif()
